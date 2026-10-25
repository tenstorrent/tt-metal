# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3b device tests: the padded last pass of the cost prefill plan (a 128-row chunk or a slab with pad rows) on the
single-slot adapter.

``pad_<form>``: the same prompt prefilled twice with two different pad token contents in the padded pass must give
bitwise-identical first-token logits and identical decode logits and tokens over 32 steps (the pad rows' content reaches
neither the recurrent state, the conv and PLE histories, the QSA staging and ring, nor the KV rows a decode reads).
``kv_<form>``: the K/V rows and compressed index rows a padded pass writes for its real rows equal, bit for bit, the same
rows of a full pass of that form (a longer prompt sharing the first tokens): the 12 QSA layers' caches depend on every
earlier layer's output for those rows, so this proves the real rows of a padded pass compute what they compute in a full
pass (a slab under the default tiled sparse attention kernel: except the real rows of the 32-row query tile that
also holds pad rows).  ``state_<form>``: the hand-off state against the legacy plan.  ``handoff_<form>``: the padded
plan against the legacy plan on the same prompt: first-token logits, decode tokens and the first divergence (a
tolerance-class comparison, not a bitwise one: the forms differ in numerics).

Run:  pytest .../test_prefill_padded_pass_device.py -v -s -k pad_long
Env:  as ``test_vllm_batch_contract_device`` (QWEN38_HARDWARE_PROFILE, TT_METAL_TRACE_ALLOC_TRACKING=1, MODEL_WEIGHTS_DIR,
      QWEN38_CACHE_ROOT, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0).  One id per process.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import initialize
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry as fused_registry
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerType

DECODE_STEPS = 32
PAD_TOKENS = (0, 151643)  # two in-vocabulary ids for the pad rows
OUT = os.environ.get("P3B_OUT")  # a directory for the per-test JSON results; unset, none are written


def save(name: str, text: str) -> None:
    if OUT:
        (Path(OUT) / name).write_text(text)


def build_prompt(checkpoint: Path, length: int) -> list[int]:
    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    render = lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text
    ids = build_prompt_ids(template.tokenizer, render, config, text, max(length, 256))[
        0
    ]  # short ones: a 256-token prompt's head
    assert len(ids) == max(length, 256)
    return ids[:length]


class Harness:
    """One adapter, a recording hook on the chain's chunk prefill and the pad token the driver uses."""

    def __init__(self, mesh_device, checkpoint: Path) -> None:
        self.model = initialize(mesh_device, checkpoint, 1)
        self.model.allocate_kv_cache((1, 1, 64, 128), None, 12)
        self.mesh = mesh_device
        self.pad_token = PAD_TOKENS[0]
        self.timing = None
        chain = self.model.chain
        original = chain.chunk_prefill

        self.capture_hand_off = False
        self.hand_off = None

        def chunk_prefill(*args, **kwargs):
            result = original(*args, **kwargs)
            self.timing = result.timing
            if self.capture_hand_off:
                self.hand_off = self.read_hand_off()
            return result

        chain.chunk_prefill = chunk_prefill
        original_init = driver_module.Qwen38ChunkPrefill.__init__
        harness = self

        def init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            self_.pad_token_id = harness.pad_token

        driver_module.Qwen38ChunkPrefill.__init__ = init

    def prefill_and_decode(
        self, prompt: list[int], *, plan: str, pad_token: int = PAD_TOKENS[0], steps=DECODE_STEPS, force=None
    ):
        """The first-token logits and ``steps`` greedy decode steps (host argmax route): ``(logits list, tokens, timing, ttft)``.
        ``force`` (a token list from another run): the decode steps consume those tokens instead of this run's own, and the
        returned tokens are this run's argmax at every step (a teacher-forced comparison of two plans)."""

        os.environ[driver_module.PLAN_ENV] = plan
        self.pad_token = pad_token
        started = time.perf_counter()
        out, _ = self.model.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(prompt)],
            start_pos=[0],
            empty_slots=[0],
        )
        ttft = time.perf_counter() - started
        timing = self.timing
        logits = [out.reshape(-1).clone()]
        tokens = [int(out.argmax())]
        for step in range(1, steps):
            out = self.model.decode_forward(
                tokens=torch.tensor([[tokens[-1] if force is None else force[step - 1]]], dtype=torch.int32),
                start_pos=[len(prompt) + step - 1],
                page_table=None,
                kv_cache=None,
                enable_trace=True,
                read_from_device=True,
            )
            logits.append(out.reshape(-1).clone())
            tokens.append(int(out.argmax()))
        return logits, tokens, timing, ttft

    def release(self) -> None:
        self.model.release_request(0)

    def read_hand_off(self) -> dict[str, list[torch.Tensor]]:
        """The decode state the hand-off just left (before the last prompt token's step), per buffer family and layer, on the
        host: GDN conv history slots and recurrent state, the PLE history slots, the QSA staging tile and raw-key ring.
        """

        chain = self.model.chain
        ttnn.synchronize_device(self.mesh)
        read = lambda tensor: [ttnn.to_torch(t).float().clone() for t in ttnn.get_device_tensors(tensor)]
        found: dict[str, list[torch.Tensor]] = {}
        for index, (layer, layer_state) in enumerate(zip(chain.built_target.model.layers, chain.state.layers)):
            attention = layer_state.attention
            if layer.layer_type is Qwen38TTNNLayerType.GDN:
                found[f"gdn_conv{index}"] = [t for slot in attention.conv_window()[:3] for t in read(slot)]
                found[f"gdn_state{index}"] = read(attention.recurrent)
            else:
                found[f"qsa_staging{index}"] = read(attention.kv_staging)
                found[f"qsa_ring{index}"] = read(attention.raw_key_ring)
            if layer_state.ple is not None:
                found[f"ple{index}"] = [t for slot in layer_state.ple.conv for t in read(slot)]
        return found

    def kv_rows(self, rows: int) -> dict[str, list[torch.Tensor]]:
        """The first ``rows`` K/V rows and ``rows // 4`` compressed index rows of every QSA layer, per device, on the host."""

        chain = self.model.chain
        layers = chain.built_target.model.layers
        state = chain.state
        ttnn.synchronize_device(self.mesh)
        found = {}
        for index, (layer, layer_state) in enumerate(zip(layers, state.layers)):
            if layer.layer_type is not Qwen38TTNNLayerType.QSA:
                continue
            attention = layer_state.attention
            found[f"kv{index}"] = [
                ttnn.to_torch(t)[..., :rows, :].clone() for t in ttnn.get_device_tensors(attention.packed_kv_cache)
            ]
            found[f"index{index}"] = [
                ttnn.to_torch(t)[..., : rows // 4, :].clone()
                for t in ttnn.get_device_tensors(attention.compressed_index_cache)
            ]
        return found


def first_divergence(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


# (id, real rows of the padded pass, rows of a full pass of that form, padded pass form)
FORMS = {
    "long": dict(real=100, full=128, kind="long"),
    "slab": dict(real=1000, full=2048, kind="slab"),
}


@pytest.fixture
def harness(mesh_device):
    checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
    mesh_device.enable_program_cache()
    built = Harness(mesh_device, checkpoint)
    try:
        yield built, checkpoint
    finally:
        built.model.release_persistent_capture()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("form", [pytest.param("long", id="pad_long"), pytest.param("slab", id="pad_slab")])
@pytest.mark.timeout(3600)
def test_pad_content_independence(mesh_device, harness, form):
    built, checkpoint = harness
    spec = FORMS[form]
    prompt = build_prompt(checkpoint, spec["real"] + 1)  # the head (all but the last token) is ``real`` rows
    runs = []
    for pad in PAD_TOKENS:
        logits, tokens, timing, ttft = built.prefill_and_decode(prompt, plan="cost", pad_token=pad)
        built.release()
        logger.info(
            f"pad token {pad}: plan {timing.plan} padded {timing.padded_pass} x {timing.padded_real_rows} real rows, "
            f"slabs {timing.slabs} long {timing.long_chunks} short {timing.chunks}, ttft {ttft:.3f} s, tokens {tokens[:8]}"
        )
        runs.append((logits, tokens, timing))
    (logits_a, tokens_a, timing_a), (logits_b, tokens_b, _) = runs
    assert timing_a.plan == "cost" and timing_a.padded_pass == spec["kind"], timing_a
    assert timing_a.padded_real_rows == spec["real"]
    first_equal = torch.equal(logits_a[0], logits_b[0])
    all_equal = all(torch.equal(a, b) for a, b in zip(logits_a, logits_b))
    logger.info(
        f"RESULT pad_{form}: first-token logits bitwise {first_equal}; decode logits bitwise {all_equal}; "
        f"tokens equal {tokens_a == tokens_b} ({len(tokens_a)} tokens)"
    )
    save(
        f"pad_{form}.json",
        json.dumps(
            dict(
                first_logits_equal=first_equal,
                all_logits_equal=all_equal,
                tokens_equal=tokens_a == tokens_b,
                tokens=tokens_a,
                timing=repr(timing_a),
            )
        ),
    )
    assert first_equal and tokens_a == tokens_b and all_equal


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("form", [pytest.param("long", id="kv_long"), pytest.param("slab", id="kv_slab")])
@pytest.mark.timeout(3600)
def test_real_rows_of_a_padded_pass_equal_a_full_pass(mesh_device, harness, form):
    built, checkpoint = harness
    spec = FORMS[form]
    full_prompt = build_prompt(checkpoint, spec["full"] + 1)
    short_prompt = full_prompt[: spec["real"]] + [full_prompt[spec["real"]]]  # the same first ``real`` head rows
    assert len(short_prompt) == spec["real"] + 1
    _, _, timing_full, _ = built.prefill_and_decode(full_prompt, plan="cost", steps=1)
    full_rows = built.kv_rows(spec["real"])
    built.release()
    assert timing_full.padded_pass is None and timing_full.plan == "cost", timing_full
    _, _, timing_pad, _ = built.prefill_and_decode(short_prompt, plan="cost", steps=1)
    pad_rows = built.kv_rows(spec["real"])
    built.release()
    assert timing_pad.padded_pass == spec["kind"] and timing_pad.padded_real_rows == spec["real"], timing_pad
    mismatches = []
    for name in full_rows:
        for device, (a, b) in enumerate(zip(full_rows[name], pad_rows[name])):
            if not torch.equal(a, b):
                bad = (a != b).reshape(a.shape[-2], -1).any(dim=1).nonzero().reshape(-1).tolist()
                mismatches.append(
                    (name, device, float((a.float() - b.float()).abs().max()), len(bad), bad[:6], bad[-3:])
                )
    logger.info(
        f"RESULT kv_{form}: {len(full_rows)} caches x {len(next(iter(full_rows.values())))} devices compared, "
        f"{len(mismatches)} mismatches {mismatches[:5]}"
    )
    save(f"kv_{form}.json", json.dumps(dict(caches=len(full_rows), mismatches=mismatches)))
    # The tiled block-shared attention kernel of the slab streams one block union per 32-row tile of queries, so the real
    # rows of the tile that also holds pad rows round with their neighbors' selections (in a full slab the next real
    # tokens'): only those rows, and their compressed index rows, may differ.  Without it every row is bitwise.
    first_shared = spec["real"] // 32 * 32
    if form == "slab" and fused_registry.enabled("sparse_sdpa_tiled"):
        first_bad = {name: (first_shared // 4 if name.startswith("index") else first_shared) for name in full_rows}
        outside = [m for m in mismatches if m[4][0] < first_bad[m[0]]]
        assert not outside, outside
    else:
        assert not mismatches


# the prompts of the hand-off comparison: real text, a head whose last pass is padded under the cost plan
HANDOFF = {"long": 256, "slab": 1001}  # head 255: 128 + a 127-row padded 128-row chunk; head 1000: one padded slab


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("form", [pytest.param("long", id="handoff_long"), pytest.param("slab", id="handoff_slab")])
@pytest.mark.timeout(3600)
def test_padded_plan_against_the_legacy_plan(mesh_device, harness, form):
    """Tolerance class: the cost plan runs the head's rows through other forms than the legacy plan.  The legacy run's
    greedy tokens are forced into both runs (the steps compare like with like): per-step top-1 agreement, the softmax KL of
    the cost run's logits against the legacy run's, and the first divergence of the two free-running greedy streams."""

    built, checkpoint = harness
    kind = FORMS[form]["kind"]
    prompt = build_prompt(checkpoint, HANDOFF[form])
    legacy = built.prefill_and_decode(prompt, plan="legacy")
    built.release()
    control = built.prefill_and_decode(
        prompt, plan="legacy", force=legacy[1]
    )  # a rerun: the noise floor (zero if exact)
    built.release()
    cost = built.prefill_and_decode(prompt, plan="cost", force=legacy[1])
    built.release()
    cost_free = built.prefill_and_decode(prompt, plan="cost")
    built.release()
    # every row through the 32-row form: what the 128-row chunk and the slab are measured against
    original_cover = driver_module.Qwen38ChunkPrefill._cover
    driver_module.Qwen38ChunkPrefill._cover = lambda self, count, position: (
        driver_module.legacy_cover(count),
        "legacy",
    )
    try:
        short_only = built.prefill_and_decode(prompt, plan="legacy", force=legacy[1])
        built.release()
    finally:
        driver_module.Qwen38ChunkPrefill._cover = original_cover

    def compare(a, b):
        agree = sum(int(x.argmax() == y.argmax()) for x, y in zip(a, b)) / len(a)
        kls = []
        for x, y in zip(a, b):
            px, py = torch.log_softmax(x.float()[:TOKENIZER_SIZE], 0), torch.log_softmax(y.float()[:TOKENIZER_SIZE], 0)
            kls.append(float((px.exp() * (px - py)).sum()))
        return agree, max(kls), sum(kls) / len(kls)

    rerun = compare(legacy[0], control[0])
    short_vs_legacy = compare(short_only[0], legacy[0])
    short_vs_cost = compare(short_only[0], cost[0])
    forced = compare(legacy[0], cost[0])
    diverge = first_divergence(legacy[1], cost_free[1])
    logger.info(
        f"RESULT handoff_{form}: prompt {len(prompt)}; legacy ttft {legacy[3]:.3f} s, cost ttft {cost_free[3]:.3f} s "
        f"(padded {cost_free[2].padded_pass} x {cost_free[2].padded_real_rows}); first divergence of the free-running "
        f"{len(cost_free[1])}-token greedy streams: {diverge}; teacher-forced top-1 agreement {forced[0]:.3f}, "
        f"KL max {forced[1]:.2e} mean {forced[2]:.2e}; legacy rerun noise floor: agreement {rerun[0]:.3f}, KL max {rerun[1]:.2e}; "
        f"32-row-only vs legacy: agreement {short_vs_legacy[0]:.3f} KL max {short_vs_legacy[1]:.2e} mean {short_vs_legacy[2]:.2e}; "
        f"32-row-only vs cost: agreement {short_vs_cost[0]:.3f} KL max {short_vs_cost[1]:.2e} mean {short_vs_cost[2]:.2e}"
    )
    save(
        f"handoff_{form}.json",
        json.dumps(
            dict(
                first_divergence=diverge,
                forced_agreement=forced[0],
                forced_kl_max=forced[1],
                forced_kl_mean=forced[2],
                rerun=rerun,
                legacy_tokens=legacy[1],
                cost_tokens=cost_free[1],
                legacy_ttft=legacy[3],
                cost_ttft=cost_free[3],
            )
        ),
    )
    assert cost_free[2].padded_pass == kind and cost_free[2].plan == "cost"


def relative_errors(a: dict, b: dict, staging_rows: int = 32) -> dict[str, float]:
    """The largest per-tensor relative L2 error ``|a - b| / |b|`` over the layers, by buffer family.  ``staging_rows``:
    the QSA staging tile ``[1,1,32,512]`` holds the prompt's open block in its first ``prefilled % 32`` rows (all 32 when the
    block is complete: the hand-off zeroes it); the rows past them are the pass's padded rows (the next replay rewrites
    them, nothing gathers them), so only the first ``staging_rows`` rows are compared."""

    worst: dict[str, float] = {}
    for name in a:
        family = name.rstrip("0123456789")
        rows = staging_rows if family == "qsa_staging" else None
        error = max(
            float((x[..., :rows, :] - y[..., :rows, :]).norm() / max(float(y[..., :rows, :].norm()), 1e-12))
            for x, y in zip(a[name], b[name])
        )
        worst[family] = max(worst.get(family, 0.0), error)
    return worst


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("form", [pytest.param("long", id="state_long"), pytest.param("slab", id="state_slab")])
@pytest.mark.timeout(3600)
def test_hand_off_state_against_the_other_plans(mesh_device, harness, form):
    """The decode state the hand-off leaves (GDN conv history and recurrent state, PLE history, QSA staging tile and
    raw-key ring, every layer) after the cost plan's padded last pass against the legacy plan's, and against the 32-row
    form alone: relative L2 error per buffer family.  A wrong row selection (a history, a kept tile) is an O(1) error; the
    forms' own numerics differ at the 1e-3 level.  Also the legacy plan against its own rerun (zero: deterministic)."""

    built, checkpoint = harness
    built.capture_hand_off = True
    prompt = build_prompt(checkpoint, HANDOFF[form])
    states = {}
    for name, plan, force_short in (
        ("legacy", "legacy", False),
        ("legacy2", "legacy", False),
        ("cost", "cost", False),
        ("short", "legacy", True),
    ):
        original_cover = driver_module.Qwen38ChunkPrefill._cover
        if force_short:
            driver_module.Qwen38ChunkPrefill._cover = lambda self, count, position: (
                driver_module.legacy_cover(count),
                "legacy",
            )
        try:
            built.prefill_and_decode(prompt, plan=plan, steps=1)
        finally:
            driver_module.Qwen38ChunkPrefill._cover = original_cover
        built.release()
        states[name] = built.hand_off
        if name == "cost":
            assert built.timing.padded_pass == FORMS[form]["kind"], built.timing
    early = {
        key: {
            name: float((x - y).norm() / max(float(y.norm()), 1e-12))
            for name, x, y in (
                (
                    f"{family}{layer}-dev{device}",
                    states[key][f"{family}{layer}"][device],
                    states["legacy"][f"{family}{layer}"][device],
                )
                for family, layer in (
                    ("gdn_conv", 0),
                    ("gdn_conv", 1),
                    ("gdn_conv", 2),
                    ("gdn_state", 0),
                    ("ple", 1),
                    ("qsa_ring", 3),
                    ("gdn_conv", 4),
                )
                for device in range(1)
            )
        }
        for key in ("cost", "short")
    }
    for key, value in early.items():
        logger.info(
            f"RESULT state_{form} early layers {key}_vs_legacy: " + ", ".join(f"{n} {e:.2e}" for n, e in value.items())
        )
    staging_rows = (
        len(prompt) - 1
    ) % 32 or 32  # the head is prefilled; its open block's rows are the real staging rows
    report = {
        "early": early,
        "cost_vs_legacy": relative_errors(states["cost"], states["legacy"], staging_rows),
        "short_vs_legacy": relative_errors(states["short"], states["legacy"], staging_rows),
        "cost_vs_short": relative_errors(states["cost"], states["short"], staging_rows),
        "legacy_vs_rerun": relative_errors(states["legacy2"], states["legacy"]),
    }
    for key, value in report.items():
        if key == "early":  # logged above, per layer
            continue
        logger.info(
            f"RESULT state_{form} {key}: " + ", ".join(f"{family} {error:.2e}" for family, error in value.items())
        )
    save(f"state_{form}.json", json.dumps(report))
    assert max(report["legacy_vs_rerun"].values()) == 0.0


ACCEPTANCE_DIR = Path(__file__).resolve().parents[1] / "tools" / "acceptance" / "greedy-prompts"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "record", [pytest.param("summary", id="record_summary"), pytest.param("math", id="record_math")]
)
@pytest.mark.timeout(3600)
def test_an_acceptance_record_whose_plan_changed(mesh_device, harness, record):
    """The plan change touches the acceptance records ``json``, ``math`` and ``summary`` (a padded 128-row chunk where the
    legacy plan ran 32-row ones).  The record's prompt through both plans on the single-slot adapter: the cost plan's greedy
    stream against the legacy plan's, the top-1 margin of the legacy run at every step (a divergence at a step with a margin
    below the plans' noise is a near-tie flip, not a defect), and the teacher-forced logit differences."""

    built, _checkpoint = harness
    prompt = json.loads((ACCEPTANCE_DIR / f"prompt-{record}-greedy.json").read_text())["prompt_token_ids"]
    legacy = built.prefill_and_decode(prompt, plan="legacy", steps=8)
    built.release()
    cost = built.prefill_and_decode(prompt, plan="cost", steps=8, force=legacy[1])
    built.release()
    steps = []
    for step, (a, b) in enumerate(zip(legacy[0], cost[0])):
        top = a.float()[:TOKENIZER_SIZE].topk(3)
        top_b = b.float()[:TOKENIZER_SIZE].topk(3)
        steps.append(
            dict(
                step=step,
                legacy_top=top.indices.tolist(),
                legacy_values=[round(float(v), 3) for v in top.values],
                cost_top=top_b.indices.tolist(),
                cost_values=[round(float(v), 3) for v in top_b.values],
                legacy_margin=float(top.values[0] - top.values[1]),
                cost_margin=float(top_b.values[0] - top_b.values[1]),
                max_abs_diff_of_top3=float((a.float()[top.indices] - b.float()[top.indices]).abs().max()),
            )
        )
    for row in steps:
        logger.info(f"RESULT record_{record} {row}")
    save(f"record_{record}.json", json.dumps(dict(prompt_tokens=len(prompt), plan=repr(cost[2]), steps=steps)))
