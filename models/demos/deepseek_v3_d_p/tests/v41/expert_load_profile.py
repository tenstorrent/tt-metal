# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Generates ``tt/v41/expert_load_profile.json``, the routed-expert load profile behind the V4.1 expert placement
(``tt/v41/expert_placement.py``, bead 8y7.9.8). CPU only (torch; no ttnn, no device)::

    python models/demos/deepseek_v3_d_p/tests/v41/expert_load_profile.py

Per prompt: the first 2048 tokens (V4.1 tokenizer, no special tokens) run through the reference stack of the
checkpoint layers (0, 2, 3, 20, 21, 24) (``oracle.oracle``, disk-cached, ~4 min per prompt cold); the reference
gate (``Gate.forward``: fp32 ``sqrt(softplus(x W^T))``, top-6 of ``scores + bias``) on each layer's MoE input
``ffn_in`` gives the routed pairs per expert. Only these layers have checkpoint shards here.

Training prompts (``TRAINING``) cover technical prose, code, legal text, Q&A, narrative prose and three non-English
languages. The held-out prompts (``HELD_OUT``: the novel of ``oracle.text_tokens`` that the production gates use, and
the vendored reference ``model.py`` as code) are never used for the placement; the device placement test runs on
them. Random token ids (the G2 profile harness input) are reported as a stress case, not trained on.

The ``validation`` field records the modelled MoE cost of the most expensive chip (``expert_placement.chip_costs``,
LoudBox 2x4) for the checkpoint order and for the placement: leave-one-out over the training prompts and on each
held-out prompt.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tt.v41 import expert_placement as P
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint

REPO = Path(__file__).resolve().parents[5]
LAYERS = (0, 2, 3, 20, 21, 24)  # every backbone layer with checkpoint shards present
SEQ = 2048
CHIPS, GROUP = 8, 2  # LoudBox 2x4: 4 dispatch groups (mesh columns) of 2 chips
EMACS = Path("/usr/share/emacs/27.1/etc/tutorials")
LICENSES = Path("/usr/share/common-licenses")


def _file(path: Path):
    return lambda: path.read_text()


def _json_prompts(path: Path):
    return lambda: "\n\n".join(item["prompt"] for item in json.loads(path.read_text()))


TRAINING = {
    "docs_llms": _file(REPO / "tech_reports/LLMs/llms.md"),
    "docs_fabric": _file(REPO / "tech_reports/TT-Fabric/TT-Fabric-Architecture.md"),
    "docs_distributed": _file(REPO / "tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md"),
    "legal_gpl3": _file(LICENSES / "GPL-3"),
    "legal_gfdl": _file(LICENSES / "GFDL-1.3"),
    "code_cpp": _file(REPO / "tt_metal/impl/program/dispatch.cpp"),
    "code_python": _file(REPO / "ttnn/ttnn/decorators.py"),
    "qa_aime_gpqa": _json_prompts(REPO / "models/demos/deepseek_v3/demo/demo_aime24_gpqa_short.json"),
    "story_25k": _json_prompts(REPO / "models/demos/deepseek_v3_d_p/demo/test_prompt_25k.json"),
    "zh_tutorial": _file(EMACS / "TUTORIAL.cn"),
    "ru_tutorial": _file(EMACS / "TUTORIAL.ru"),
    "de_tutorial": _file(EMACS / "TUTORIAL.de"),
}
HELD_OUT = {
    "novel": None,  # oracle.text_tokens
    "code_reference": _file(REPO / "models/demos/deepseek_v3_d_p/reference/deepseek_v41/model.py"),
}


def spec() -> orc.OracleSpec:
    return orc.real_spec(LAYERS, SEQ, candidate_topk_blocks=96, checkpoint=orc.HF_SNAPSHOT)


def prompt_tokens(name: str) -> torch.Tensor:
    """[1, SEQ] tokens of a training or held-out prompt."""
    if name == "novel":
        return orc.text_tokens(SEQ)
    if name == "random":
        return orc.random_tokens(spec())
    from transformers import AutoTokenizer

    text = {**TRAINING, **HELD_OUT}[name]()
    ids = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= SEQ, (name, len(ids))
    return torch.tensor(ids[:SEQ], dtype=torch.int64)[None]


def gate_weights(layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    names = [f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"]
    w = resolve_checkpoint().read(names)
    return w[names[0]].float(), w[names[1]].float()


def reference_indices(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The reference ``Gate`` selection (text tokens) for the MoE inputs ``x`` [n, dim]."""
    scores = F.softplus(x.float() @ weight.T).sqrt()  # score_func sqrtsoftplus, gate_temp 1
    return (scores + bias).topk(C.NUM_EXPERTS_PER_TOKEN, dim=-1)[1]


def prompt_counts(name: str, gates: dict) -> dict[int, list[int]]:
    tokens = prompt_tokens(name)
    start = time.perf_counter()
    result = orc.oracle(spec(), tokens)
    logger.info(f"{name}: oracle {time.perf_counter() - start:.1f}s")
    return {
        layer: torch.bincount(
            reference_indices(result["blocks"][layer]["ffn_in"], *gates[layer]).flatten(),
            minlength=C.NUM_ROUTED_EXPERTS,
        ).tolist()
        for layer in LAYERS
    }


def max_chip_ms(counts: np.ndarray, chip) -> float:
    """Most expensive chip (ms) of the placement cost model for one prompt's routed pairs at the chunk scale."""
    return float(P.chip_costs(counts, np.asarray(chip), GROUP, CHIPS).max() / 1e3)


def main() -> None:
    torch.set_num_threads(8)
    gates = {layer: gate_weights(layer) for layer in LAYERS}
    names = list(TRAINING) + list(HELD_OUT) + ["random"]
    counts = {name: prompt_counts(name, gates) for name in names}
    scale = P.CHUNK_TOKENS / SEQ
    identity = [e // (C.NUM_ROUTED_EXPERTS // CHIPS) for e in range(C.NUM_ROUTED_EXPERTS)]
    validation = {}
    for layer in LAYERS:
        train = np.asarray([counts[n][layer] for n in TRAINING], dtype=np.float64) * scale
        chip = P.place(train, GROUP, CHIPS // GROUP)
        loo = [
            (
                max_chip_ms(train[i], identity),
                max_chip_ms(train[i], P.place(np.delete(train, i, 0), GROUP, CHIPS // GROUP)),
            )
            for i in range(len(train))
        ]
        row = {"leave_one_out": [round(float(np.mean([v[k] for v in loo])), 3) for k in (0, 1)]}
        for n in list(HELD_OUT) + ["random"]:
            c = np.asarray(counts[n][layer], dtype=np.float64) * scale
            row[n] = [round(max_chip_ms(c, identity), 3), round(max_chip_ms(c, chip), 3)]
        validation[str(layer)] = row
        logger.info(f"L{layer} modelled most expensive chip ms [checkpoint order, placed]: {row}")
    profile = {
        "source": {
            "description": "routed (token, expert) pairs per expert of the reference gate, first 2048 tokens per "
            "prompt, reference stack of the checkpoint layers (0, 2, 3, 20, 21, 24); generated by "
            "tests/v41/expert_load_profile.py",
            "checkpoint": str(orc.HF_SNAPSHOT.name),
            "seq_len": SEQ,
            "prompts": {n: hashlib.sha256(prompt_tokens(n).numpy().tobytes()).hexdigest()[:16] for n in TRAINING},
        },
        "validation": {
            "metric": f"modelled MoE cost of the most expensive chip (ms, expert_placement.chip_costs, {P.CHUNK_TOKENS}-token "
            f"chunk, {GROUP}x{CHIPS // GROUP} mesh), [checkpoint order, placed]",
            "held_out": {n: hashlib.sha256(prompt_tokens(n).numpy().tobytes()).hexdigest()[:16] for n in HELD_OUT},
            "layers": validation,
        },
        "counts": {str(layer): [counts[n][layer] for n in TRAINING] for layer in LAYERS},
    }
    counts_json = profile.pop("counts")
    rows = ",\n".join(
        f'  "{layer}": [\n' + ",\n".join(f"   {json.dumps(r)}" for r in layer_rows) + "\n  ]"
        for layer, layer_rows in counts_json.items()
    )
    text = json.dumps(profile, indent=1)[:-2] + ',\n "counts": {\n' + rows + "\n }\n}\n"
    assert json.loads(text)["counts"] == counts_json
    tmp = P.PROFILE_PATH.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(P.PROFILE_PATH)
    logger.info(f"wrote {P.PROFILE_PATH}")


if __name__ == "__main__":
    main()
