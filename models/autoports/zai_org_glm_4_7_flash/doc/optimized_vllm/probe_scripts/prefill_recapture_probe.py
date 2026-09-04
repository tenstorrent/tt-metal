# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Why does every served prefill trigger a decode-trace recapture?

The 100/100/32 CI serving burst recaptures the decode traces once per admitted
request (server.log: one "Resetting sampling trace" per prefill, ~0.52 s apart
before this stage's change, ~1.32 s after it because there are now four decode
traces instead of one). ``_maybe_recapture_after_compile`` only fires when the
program cache grew while the traces were live, so something in the served
prefill path compiles a new program on *every* call even at a warmed prompt
length. This probe isolates which call it is, on the reduced 2-layer model.

    python .../probe_scripts/prefill_recapture_probe.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))

from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import GLM47FlashForCausalLM  # noqa: E402
from models.common.sampling import SamplingParams  # noqa: E402

MODEL_DIR = REPO / "models" / "autoports" / "zai_org_glm_4_7_flash"
#: ``before`` reproduces the pre-stage arm by turning the per-slot prefill warm
#: off; the default writes the ``after`` arm. Both committed files are
#: therefore re-runnable, which the unsuffixed name was not.
_ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
OUT = MODEL_DIR / "doc" / "optimized_vllm" / f"prefill_recapture_probe_{_ARM}.json"

MAX_SEQ_LEN = 4096
BLOCK_SIZE = 64
MAX_BATCH = 32
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
NUM_BLOCKS = MAX_BATCH * BLOCKS_PER_USER
PROMPT_LEN = 100  # exactly the CI serving-burst prompt length


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    report = {}
    try:
        gen = build_generator(
            MODEL_DIR,
            dev,
            layer_indices=[0, 1],
            max_batch_size=MAX_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            defer_cache_and_traces=True,
            prefill_slot_warmup=(_ARM != "before"),
            progress=lambda m: None,
        )
        model = GLM47FlashForCausalLM(gen)
        kv = model.allocate_kv_cache(
            kv_cache_shape=(NUM_BLOCKS, 1, BLOCK_SIZE, gen.model.layers[0].kvpe_dim),
            dtype=torch.bfloat16,
            num_layers=len(gen.model.layers),
        )
        model.warmup_model_prefill(kv_cache=kv, can_sample_on_device=True, enable_trace=True)
        model.warmup_model_decode(
            kv_cache=kv, max_batch_size=MAX_BATCH, num_blocks=NUM_BLOCKS, can_sample_on_device=True, enable_trace=True
        )
        gen.reset()

        def entries():
            return dev.num_program_cache_entries()

        params = SamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0])
        prompt = list(range(1000, 1000 + PROMPT_LEN))
        steps = []
        for slot in range(6):
            e0 = entries()
            gen.apply_prefill_sampling_state(params, empty_slots=[slot])
            e1 = entries()
            gen.prefill_and_sample(prompt, user_id=slot, recapture=False)
            e2 = entries()
            recaptured = gen._maybe_recapture_after_compile()
            e3 = entries()
            steps.append(
                {
                    "slot": slot,
                    "entries_start": e0,
                    "compiled_by_apply_prefill_sampling_state": e1 - e0,
                    "compiled_by_prefill_and_sample": e2 - e1,
                    "recaptured": bool(recaptured),
                    "compiled_by_recapture": e3 - e2,
                }
            )
            print(steps[-1], flush=True)

        # Same prompt length, same slot, repeated: does it ever settle?
        settle = []
        for i in range(4):
            e0 = entries()
            gen.apply_prefill_sampling_state(params, empty_slots=[0])
            gen.prefill_and_sample(prompt, user_id=0, recapture=False)
            e1 = entries()
            recaptured = gen._maybe_recapture_after_compile()
            settle.append({"iter": i, "compiled": e1 - e0, "recaptured": bool(recaptured)})
            print(settle[-1], flush=True)

        # Which op is it? Forbid cache misses and let the first unseen slot name it.
        offender = None
        try:
            dev.set_program_cache_misses_allowed(False)
            gen.apply_prefill_sampling_state(params, empty_slots=[20])
            gen.prefill_and_sample(prompt, user_id=20, recapture=False)
            offender = "none (no miss on a fresh slot)"
        except Exception as exc:  # noqa: BLE001
            offender = str(exc).strip().splitlines()[0][:400]
        finally:
            dev.set_program_cache_misses_allowed(True)
        print("OFFENDER:", offender, flush=True)
        gen._maybe_recapture_after_compile()

        # Does the per-slot program depend on prompt length too, or only on the slot?
        length_cross = []
        for plen in (100, 200, 400):
            e0 = entries()
            gen.apply_prefill_sampling_state(params, empty_slots=[7])
            gen.prefill_and_sample(list(range(1000, 1000 + plen)), user_id=7, recapture=False)
            length_cross.append({"slot": 7, "prompt_len": plen, "compiled": entries() - e0})
            gen._maybe_recapture_after_compile()
            print(length_cross[-1], flush=True)

        report = {
            "offending_op": offender,
            "slot_program_vs_prompt_length": length_cross,
            "purpose": (
                "Attribute the per-served-prefill decode-trace recapture to the exact call that grows the "
                "program cache, on the reduced 2-layer model at the CI serving-burst prompt length."
            ),
            "prompt_len": PROMPT_LEN,
            "per_slot": steps,
            "repeat_same_slot": settle,
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print("WROTE", OUT, flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
