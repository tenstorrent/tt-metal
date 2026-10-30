# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for Call 1 (text generation).

Real input (Source-A tokenizer, 32 distinct prompts) -> the SAME chained
pipeline the demo runs (`tt/pipeline.py`) -> real task output, asserted against
the HF golden.

  Gate 1  every routed graduated stub is still native ttnn, and every sharded
          body keeps its ShardTensor2dMesh + all_reduce (a TP=2 body is NOT
          rewritten to replication)
  Gate 2  all ten graduated modules actually executed inside the forward path
  Gate 3  final-output PCC vs the HF golden >= 0.95, for every one of the 32
          samples against its OWN golden row

Run:  ./python_env/bin/python -m pytest \
        models/demos/nvidia_nemotron_3_5_lightning_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tests.e2e import make_golden
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _invocation
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

DEMO_DIR = Path(P.__file__).resolve().parents[1]
STUBS = DEMO_DIR / "_stubs"

PCC_THRESHOLD = 0.95

# Depth actually built by this gate. Forced by DRAM (see README / tt/_hf_ref.py):
# the 23 MoE blocks alone need ~29 GB per chip at TP=2 against ~12 GB available.
GATE_LAYERS = int(os.environ.get("TT_E2E_LAYERS", P.DEFAULT_LAYERS))
# Batch is READ FROM THE PIPELINE, never typed into an assertion message.
GATE_BATCH = int(os.environ.get("TT_E2E_BATCH", P.BATCH))

# Host-side torch COMPUTE ops that must not run in a stub's forward.
FORBIDDEN_TORCH_FNS = {
    "matmul",
    "mm",
    "bmm",
    "einsum",
    "softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "batch_norm",
    "group_norm",
    "embedding",
    "embedding_bag",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "scaled_dot_product_attention",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "dropout",
    "argmax",
    "topk",
    "multinomial",
}
FORBIDDEN_HF = re.compile(r"\.generate\(|\.forward\s*=")
FORBIDDEN_SWEEP = re.compile(r"def\s+(coverage_step|coverage_sweep|invoke_all_stubs|_touch_all_graduated)\b")

# Functions that run on EVERY call -- the hot path the contract governs.
HOT_FNS = ("__call__", "forward", "_mix", "_route", "decode_step", "decode_prefill")
HOT_PREFIXES = ("_apply_", "run_", "_trace_step")

# One-time constant builders. They are memoised and only ever execute on the
# FIRST call for a given shape; the trace contract primes them from
# `<stage>_trace_setup`, i.e. outside the captured region. Building a constant
# with torch is explicitly allowed prep, not forward compute.
PREP_FNS = {"__init__", "_ensure_consts", "_ensure_seq", "_get_consts", "build", "_get_causal_mask"}


def _torch_compute_calls(src: str, hot_only: bool = True):
    """AST-walk `src` and yield forbidden torch COMPUTE calls in hot functions.

    AST rather than regex: the stubs quote the HF reference (`F.linear(...)`,
    `torch.einsum`) in their docstrings, and a text scan reports those as
    violations that do not exist in the code.
    """
    import ast

    tree = ast.parse(src)
    hits = []

    def is_hot(name):
        return name in HOT_FNS or any(name.startswith(p) for p in HOT_PREFIXES)

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in PREP_FNS:
            continue
        if hot_only and not is_hot(fn.name):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = dotted(node.func)
            if not name:
                continue
            if name.startswith("F.") or name.startswith("torch.nn.functional."):
                hits.append((fn.name, name))
            elif name.startswith("torch.") and name.split(".")[-1] in FORBIDDEN_TORCH_FNS:
                hits.append((fn.name, name))
    return hits


# --------------------------------------------------------------------------- #
#  session fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    dev = P.open_mesh(2, 2)
    yield dev
    P.close_mesh(dev)


@pytest.fixture(scope="module")
def pipe(device):
    os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")
    t0 = time.time()
    p = P.build_pipeline(device, layers=GATE_LAYERS, batch=GATE_BATCH)
    print(f"\n[e2e] built depth={p.n_layers} batch={p.batch} in {time.time() - t0:.1f}s")
    print(f"[e2e] block_types={p.block_types}")
    print(f"[e2e] variants={p.variants}")
    return p


@pytest.fixture(scope="module")
def run(pipe):
    """ONE on-device run of the real pipeline, shared by every gate below."""
    input_ids = make_golden.build_input_ids(pipe.batch)
    _invocation.reset()
    t0 = time.time()
    tt = P.NemotronHPipeline.run_text_generation(pipe, input_ids)
    dt = time.time() - t0
    print(f"[e2e] TT decode: {tt['steps']} steps x {pipe.batch} samples in {dt:.1f}s")
    invoked = _invocation.snapshot()

    t0 = time.time()
    ref = pipe._hf_reference_text_generation(input_ids, max_new_tokens=tt["steps"])
    print(f"[e2e] HF golden (free-running generate): {ref['steps']} steps in {time.time() - t0:.1f}s")

    tf = pipe._hf_reference_teacher_forced(tt["sequences"], input_ids.shape[1], tt["steps"])
    print(f"[e2e] HF golden (same-prefix): {tuple(tf.shape)}")
    return {"input_ids": input_ids, "tt": tt, "ref": ref, "tf": tf, "invoked": invoked}


# --------------------------------------------------------------------------- #
#  Gate 1 -- still real ttnn, still sharded
# --------------------------------------------------------------------------- #
def _routed_stub_files():
    return {n: STUBS / f"{n}.py" for n in P.GRADUATED_MODULES}


def test_graduated_set_matches_bringup():
    """S1/S2: the set this pipeline routes IS the set bring-up graduated."""
    on_disk = set()
    for snap in list(STUBS.glob("*.py.last_good_native")) + list(STUBS.glob("*.py.last_good_sharded")):
        on_disk.add(snap.name.split(".py.")[0])
    assert on_disk == set(
        P.GRADUATED_MODULES
    ), f"graduated-set drift: on disk {sorted(on_disk)} vs routed {sorted(P.GRADUATED_MODULES)}"


def test_gate1_stubs_are_native_ttnn():
    """No host torch compute op and no coverage sweep in any routed stub's forward."""
    offenders = []
    for name, path in _routed_stub_files().items():
        src = path.read_text()
        for fn, call in _torch_compute_calls(src):
            offenders.append(f"{name}.{fn}: torch compute op -> {call}")
        if FORBIDDEN_SWEEP.search(src):
            offenders.append(f"{name}: defines a coverage sweep")
    assert not offenders, "Gate 1 violated:\n" + "\n".join(offenders)


def test_gate1_pipeline_has_no_shortcut():
    src = (DEMO_DIR / "tt" / "pipeline.py").read_text()
    assert not FORBIDDEN_SWEEP.search(src), "pipeline defines a coverage sweep"

    offenders = [f"{fn}: {call}" for fn, call in _torch_compute_calls(src)]
    assert not offenders, "pipeline hot path uses torch compute:\n" + "\n".join(offenders)

    # HF orchestration: allowed ONLY inside the golden helper and trace setup.
    # AST-based, because the module docstring legitimately says the words
    # "model.generate()" while promising NOT to call it.
    import ast

    allowed = {"_hf_reference_text_generation", "prefill_trace_setup", "decode_trace_setup"}
    bad = []
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef) or fn.name in allowed:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "generate":
                bad.append(f"{fn.name}: .generate()")
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == "forward":
                        bad.append(f"{fn.name}: monkey-patches .forward")
    assert not bad, "pipeline uses HF orchestration outside the golden helper:\n" + "\n".join(bad)


def test_gate1_sharded_bodies_kept_their_sharding():
    """No graduated body may LOSE sharding it had in its snapshot.

    The rule is "did it regress", not "does it shard": `nemotron_h_topk_router`
    is REPLICATED by design even in its sharded snapshot (a 2688x128 gate whose
    full-width logits every chip needs), so demanding a collective of it would
    be demanding a split that would be wrong.
    """
    bad = []
    for name in P.GRADUATED_MODULES:
        snap = STUBS / f"{name}.py.last_good_sharded"
        if not snap.exists():
            continue
        was, live = snap.read_text(), (STUBS / f"{name}.py").read_text()
        for marker in ("ShardTensor2dMesh", "ShardTensorToMesh", "all_reduce", "all_gather"):
            if marker in was and marker not in live:
                bad.append(f"{name}: lost {marker}")
    assert not bad, "sharded stubs rewritten to replication:\n" + "\n".join(bad)


def test_gate1_sharding_is_live_on_device(pipe):
    ev = pipe.sharding_evidence()
    print(f"[gate1] sharded stub instances: {ev}")
    assert pipe.sharded, "pipeline did not take its sharded branch"
    assert ev, "no built stub took its TP-sharded branch -- this is a pure-replication pipeline"


# --------------------------------------------------------------------------- #
#  Gate 2 -- every graduated module invoked in the real forward path
# --------------------------------------------------------------------------- #
def test_gate2_all_graduated_modules_invoked(run):
    invoked = run["invoked"]
    print(f"[gate2] invoked ({len(invoked)}): {sorted(invoked)}")
    missing = set(P.GRADUATED_MODULES) - invoked
    assert not missing, f"graduated modules never invoked: {sorted(missing)}"


# --------------------------------------------------------------------------- #
#  Gate 3 -- PCC vs the HF golden
# --------------------------------------------------------------------------- #
def test_gate3_e2e_pcc(run, pipe):
    """Gate 3.

    ASSERTED metric: per-step next-token-logit PCC against the HF golden
    evaluated on the SAME prefix the TT pipeline was in at that step, minimum
    over the 32 samples. That is what the port is responsible for: given a
    context, produce the right distribution. The TT path is never fed a
    reference tensor -- every step still consumes the previous TT step's own
    output; only the golden is evaluated on TT's history.

    ALSO REPORTED (not asserted): the free-running comparison against
    `model.generate()`. At the DRAM-forced 7-block depth the argmax is
    genuinely degenerate -- see `test_gate3_report_freerunning_divergence`,
    which measures it -- so free-running sequence agreement scores the
    truncation's conditioning, not the port.
    """
    tt, ref, tf = run["tt"], run["ref"], run["tf"]
    steps = tt["steps"]
    a = tt["step_logits"][:steps]  # (steps, B, vocab)
    B = a.shape[1]
    assert B == pipe.batch, f"ran {B} samples, pipeline says {pipe.batch}"

    per_sample = [P.pcc(a[:, i, :], tf[:, i, :]) for i in range(B)]
    per_step = [P.pcc(a[s], tf[s]) for s in range(steps)]
    step0 = [P.pcc(a[0, i, :], tf[0, i, :]) for i in range(B)]

    # a pipeline that shape-supports B but emits 32 identical rows is WRONG
    distinct = len({tuple(r) for r in tt["new_ids"].tolist()})

    print(f"[gate3] steps={steps} batch={B} distinct_tt_outputs={distinct}")
    print(f"[gate3] same-prefix per-step PCC : {[round(x, 5) for x in per_step]}")
    print(f"[gate3] same-prefix per-sample   : min={min(per_sample):.6f} max={max(per_sample):.6f}")
    print(f"[gate3] step-0 (identical prompt): min={min(step0):.6f} max={max(step0):.6f}")

    fr = [P.pcc(a[s], ref["step_logits"][s]) for s in range(min(steps, ref["steps"]))]
    fr_agree = (tt["new_ids"][:, :steps] == ref["new_ids"][:, :steps]).float().mean().item()
    print(f"[gate3] free-running per-step PCC: {[round(x, 5) for x in fr]}")
    print(f"[gate3] free-running token agreement vs generate(): {fr_agree * 100:.1f}%")

    achieved_pcc = min(per_sample)
    print(f"e2e PCC={achieved_pcc}")
    assert distinct > 1, f"all {B} samples produced identical output -- the batch axis is fake"
    assert achieved_pcc >= PCC_THRESHOLD, (
        f"e2e PCC {achieved_pcc} < {PCC_THRESHOLD} (worst sample "
        f"{per_sample.index(achieved_pcc)}); per-step {per_step}"
    )


def test_gate3_report_freerunning_divergence(run, pipe):
    """Measure WHY free-running greedy decode diverges: near-ties, or real error?

    For every sample, compare the HF golden's own top-1/top-2 logit gap where TT
    agrees with it against where TT disagrees. If the disagreements sit where the
    reference itself has no meaningful preference, the divergence is the
    truncated model's degeneracy rather than a port defect. This test REPORTS;
    it only fails if disagreements happen on CONFIDENT rows, which would be a
    genuine defect.
    """
    tt, tf = run["tt"], run["tf"]
    a = tt["step_logits"][0].double()  # step 0: both sides on the identical prompt
    g = tf[0].double()
    B = a.shape[0]

    top2 = g.topk(2, dim=-1)
    gap = (top2.values[:, 0] - top2.values[:, 1]) / g.std(dim=-1)
    agree = a.argmax(-1) == g.argmax(-1)

    print(f"[divergence] step-0 argmax agreement: {agree.float().mean().item() * 100:.1f}%")
    for lbl, m in (("agree", agree), ("disagree", ~agree)):
        if m.any():
            print(
                f"[divergence]   {lbl:9s} n={int(m.sum()):2d}  golden top1-top2 gap "
                f"(sigma): median={gap[m].median().item():.4f} max={gap[m].max().item():.4f}"
            )

    confident_misses = int(((~agree) & (gap > 1.0)).sum())
    print(f"[divergence] disagreements on CONFIDENT rows (gap > 1 sigma): {confident_misses}")
    assert confident_misses == 0, (
        f"{confident_misses} sample(s) disagree with the golden where the golden is confident "
        "(>1 sigma) -- that is a port defect, not truncation degeneracy"
    )


def test_batch_row0_matches_unbatched(pipe):
    """S9: row 0 of the B=N run must equal a B=1 run -- proves no sample is
    silently dropped by a hard-coded leading 1 in a stub's slice bounds."""
    ids = make_golden.build_input_ids(pipe.batch)
    big = P._first_shard(pipe.forward_logits(pipe._ids_to_device(ids), last_only=True))
    one = P._first_shard(pipe.forward_logits(pipe._ids_to_device(ids[:1]), last_only=True))
    p = P.pcc(big.reshape(pipe.batch, -1)[0], one.reshape(1, -1)[0])
    print(f"[batch] PCC(row0 of B={pipe.batch}, B=1 run) = {p:.6f}")
    assert p >= 0.999, "batching changes row 0 -- a stub is dropping samples"


# --------------------------------------------------------------------------- #
#  demo/test share ONE pipeline
# --------------------------------------------------------------------------- #
def test_demo_and_test_share_one_pipeline():
    src = (DEMO_DIR / "demo" / "demo_text_generation.py").read_text()
    assert "run_text_generation" in src and "from models.demos" in src
    assert "for layer in self.layers" not in src, "demo re-implements the layer loop"
    assert "_stubs" not in src, "demo builds stubs itself instead of using tt/pipeline.py"
