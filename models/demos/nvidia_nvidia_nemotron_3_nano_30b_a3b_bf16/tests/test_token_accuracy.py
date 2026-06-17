# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Token-accuracy regression test for NemotronH-30B (TP=4 QB).

Validates that the model produces numerically consistent predictions by
comparing teacher-forced top-1/top-5 predictions against a saved reference.

Workflow
--------
1. First run (or with --generate-ref): run greedy autoregressive decode from
   a fixed prompt; record the top-5 logits at every position; save to a .refpt
   file under tests/reference_outputs/.
2. Subsequent runs (default): load the .refpt, run teacher-forcing (feed
   reference tokens, not the model's own predictions), compute top-1 / top-5
   match rate against the saved reference, assert thresholds.

Teacher-forcing detects silent regressions:
  - DRAM-address corruption → wrong weight values → wrong top-1
  - SSM state propagation bugs → drift in predictions
  - Numerical precision changes (e.g. prewarm, L1-upload changes)

Because the reference itself comes from the TT model (not a CPU reference),
this is an *internal consistency* check rather than an absolute quality gate.
But it is strong: a healthy model should reproduce its own top-5 predictions
with high fidelity under teacher forcing.

Thresholds (conservative to allow for bfloat16 jitter and non-determinism):
  TOP1_THRESHOLD = 0.80   — 80% of positions: our top-1 == reference top-1
  TOP5_THRESHOLD = 0.95   — 95% of positions: our top-1 ∈ reference top-5

Usage
-----
    # One-time: generate reference (requires loaded model)
    pytest tests/test_token_accuracy.py -k test_generate_reference -s

    # Regression test
    pytest tests/test_token_accuracy.py -k test_token_accuracy -s
"""
from __future__ import annotations

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_outputs")
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "nemotron30b_bf16_tp4.refpt")

TOP1_THRESHOLD = 0.80
TOP5_THRESHOLD = 0.95

# N_TOKENS: total sequence length for the reference run.
# first N_TOKENS//2  = prompt fed as teacher-forcing input
# last  N_TOKENS//2  = tokens we validate predictions for
N_TOKENS = 200

# Fixed validation prompt (content-diverse to exercise all layer types).
REFERENCE_PROMPT = (
    "The Tenstorrent Blackhole architecture is a high-performance chip designed "
    "for machine learning acceleration.  It features a tile-based dataflow "
    "execution model where each Tensix core handles its own dispatch queue.  "
    "NemotronH-30B is a 52-layer hybrid model combining Mamba2 SSM layers "
    "with sparse MoE transformer blocks.  The model was trained on a diverse "
    "mix of text data including code, mathematics, and natural language.  "
    "Inference on the Blackhole QuietBox uses four chips in tensor-parallel "
    "mode (TP=4) with FABRIC_1D CCL topology.  The SSM state for each Mamba2 "
    "layer has shape [1, 64, 64, 128] and is updated in-place every token.  "
    "Paged KV caches serve the six dense-attention layers."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_model(request):
    """Return (mesh, wc) from the pytest fixture or create lazily."""
    if hasattr(request, "param") and request.param is not None:
        return request.param
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import open_device_tp4

    return open_device_tp4(), WeightCache()


def _load_tokenizer():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import _load_tokenizer as _lt

    return _lt()


def _run_forward_single(mesh, wc, ids_tt, state, pos: int, cpu_gate: bool = True):
    """One forward pass at `pos`; returns logits on CPU [1, 1, vocab]."""
    import ttnn
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import _update_ids, _update_pos
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    _update_ids(ids_tt, int(ids_tt.shape[0]))  # dummy; caller must set ids_tt before calling
    _update_pos(state.current_pos, pos)
    logits_tt = nemotron_h_forward_stateful(mesh, ids_tt, wc, state, cpu_gate=cpu_gate)
    ttnn.synchronize_device(mesh)
    return _host_rep(logits_tt, mesh, 1)  # [1, 1, vocab]


def run_teacher_forcing(mesh, wc, input_ids: list[int], n_eval: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model in teacher-forcing mode over `input_ids`.

    At each position, feed `input_ids[pos]`, get logits, record top-5 predictions.
    The model state is advanced using the *reference* token at each step (not its
    own prediction), so any divergence from the reference run is detectable.

    Args:
        mesh:       Open MeshDevice.
        wc:         WeightCache.
        input_ids:  Full token sequence (prompt + reference decoded tokens).
                    Length must be >= n_eval + 1.
        n_eval:     Number of tail positions to evaluate (second half of sequence).

    Returns:
        predicted_top1: [n_eval] int64 — model's top-1 prediction at each eval position
        predicted_top5: [n_eval, 5] int64 — model's top-5 predictions at each eval position
    """
    import ttnn
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import (
        _to_device_token,
        _update_ids,
        _update_pos,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.kv_cache import (
        DEFAULT_MAX_SEQ_LEN,
        allocate_decoder_state,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    n_total = len(input_ids)
    assert n_total >= n_eval + 1, f"Need at least {n_eval + 1} tokens, got {n_total}"

    max_seq = max(DEFAULT_MAX_SEQ_LEN, n_total + 10)
    state = allocate_decoder_state(mesh, B=1, max_seq_len=max_seq)
    ids_tt = _to_device_token(input_ids[0], mesh)

    eval_start = n_total - n_eval  # first position counted in accuracy

    top1_list = []
    top5_list = []

    for pos in range(n_total - 1):
        tok = input_ids[pos]
        _update_ids(ids_tt, tok)
        _update_pos(state.current_pos, pos)
        logits_tt = nemotron_h_forward_stateful(mesh, ids_tt, wc, state, cpu_gate=True)
        ttnn.synchronize_device(mesh)
        state.advance()

        if pos >= eval_start:
            logits_cpu = _host_rep(logits_tt, mesh, 1)[0, 0].float()  # [vocab]
            t5 = torch.topk(logits_cpu, 5).indices  # [5]
            top1_list.append(t5[0].unsqueeze(0))
            top5_list.append(t5.unsqueeze(0))

    predicted_top1 = torch.cat(top1_list, dim=0)  # [n_eval]
    predicted_top5 = torch.cat(top5_list, dim=0)  # [n_eval, 5]
    return predicted_top1, predicted_top5


# ---------------------------------------------------------------------------
# Reference generation
# ---------------------------------------------------------------------------


def generate_reference(mesh, wc) -> None:
    """Run greedy decode from REFERENCE_PROMPT; save top-5 logits to REFERENCE_FILE.

    Called once by the ``test_generate_reference`` test or manually.
    """
    import ttnn
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import (
        _to_device_token,
        _update_ids,
        _update_pos,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.kv_cache import (
        DEFAULT_MAX_SEQ_LEN,
        allocate_decoder_state,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    tokenizer = _load_tokenizer()
    prompt_ids = tokenizer.encode(REFERENCE_PROMPT, add_special_tokens=True)
    n_generate = max(0, N_TOKENS - len(prompt_ids))
    max_seq = max(DEFAULT_MAX_SEQ_LEN, N_TOKENS + 20)

    state = allocate_decoder_state(mesh, B=1, max_seq_len=max_seq)
    ids_tt = _to_device_token(prompt_ids[0], mesh)

    all_tokens = list(prompt_ids)
    all_top5: list[torch.Tensor] = []

    print(f"[ref] Prompt: {len(prompt_ids)} tokens, generating {n_generate} more → total {N_TOKENS}", flush=True)

    n_total = len(prompt_ids) + n_generate
    next_tok = prompt_ids[0]

    for pos in range(n_total - 1):
        tok = all_tokens[pos] if pos < len(all_tokens) else next_tok
        _update_ids(ids_tt, tok)
        _update_pos(state.current_pos, pos)
        logits_tt = nemotron_h_forward_stateful(mesh, ids_tt, wc, state, cpu_gate=True)
        ttnn.synchronize_device(mesh)
        state.advance()

        logits_cpu = _host_rep(logits_tt, mesh, 1)[0, 0].float()  # [vocab]
        t5 = torch.topk(logits_cpu, 5).indices  # [5]
        all_top5.append(t5.unsqueeze(0))

        if pos >= len(prompt_ids) - 1:
            next_tok = int(t5[0])
            all_tokens.append(next_tok)
            if tokenizer.eos_token_id is not None and next_tok == tokenizer.eos_token_id:
                print(f"[ref] EOS at position {pos + 1}", flush=True)
                break

        if (pos + 1) % 20 == 0:
            print(f"[ref] {pos + 1}/{n_total - 1} tokens processed", flush=True)

    reference_tokens = torch.tensor(all_tokens, dtype=torch.int64).unsqueeze(0)  # [1, N]
    top5_tokens = torch.cat(all_top5, dim=0)  # [N-1, 5]

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    torch.save({"reference_tokens": reference_tokens, "top5_tokens": top5_tokens}, REFERENCE_FILE)
    print(f"[ref] Saved: {REFERENCE_FILE}  (tokens={reference_tokens.shape}, top5={top5_tokens.shape})", flush=True)


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------


def compute_accuracy(
    predicted_top1: torch.Tensor,  # [n_eval]
    predicted_top5: torch.Tensor,  # [n_eval, 5]
    ref_top5: torch.Tensor,  # [n_eval, 5]
) -> tuple[float, float]:
    """Return (top1_accuracy, top5_accuracy).

    top1_accuracy: fraction where our top-1 == reference top-1.
    top5_accuracy: fraction where our top-1 ∈ reference top-5.
    """
    n = predicted_top1.shape[0]
    ref_top1 = ref_top5[:, 0]  # [n_eval]

    top1_matches = (predicted_top1 == ref_top1).sum().item()

    # our_top1 in ref top-5 at each position
    our_1 = predicted_top1.unsqueeze(1).expand(-1, 5)  # [n, 5]
    top5_matches = (our_1 == ref_top5).any(dim=1).sum().item()

    return top1_matches / n, top5_matches / n


# ---------------------------------------------------------------------------
# Pytest fixtures (optional hardware fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_handles():
    """Open device + load weights once per module; yield (mesh, wc); close after."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    mesh = open_device_tp4()
    wc = WeightCache()
    yield mesh, wc
    close_device_tp4(mesh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_reference(model_handles):
    """Generate and save the .refpt reference file.  Run once to create it.

    This test is intentionally skipped if the file already exists — use
    ``--generate-ref`` in the CLI or delete the file to regenerate.
    """
    if os.path.exists(REFERENCE_FILE):
        pytest.skip(f"Reference already exists at {REFERENCE_FILE}; delete to regenerate.")
    mesh, wc = model_handles
    generate_reference(mesh, wc)
    assert os.path.exists(REFERENCE_FILE), "Reference file was not created"


@pytest.mark.slow
def test_token_accuracy(model_handles):
    """Teacher-forcing regression test.

    Feeds the saved reference tokens through the model one-by-one and verifies
    that the model's predictions match the saved reference to within threshold.

    Fails if:
      - top-1 accuracy < TOP1_THRESHOLD (0.80)
      - top-5 accuracy < TOP5_THRESHOLD (0.95)
    """
    if not os.path.exists(REFERENCE_FILE):
        pytest.skip(f"No reference file at {REFERENCE_FILE}. " "Run test_generate_reference first to create it.")

    ref = torch.load(REFERENCE_FILE, weights_only=False)
    reference_tokens = ref["reference_tokens"]  # [1, N]
    ref_top5 = ref["top5_tokens"]  # [N-1, 5]

    n_total = reference_tokens.shape[1]
    split_point = n_total // 2
    n_eval = n_total - split_point - 1  # positions [split_point .. n_total-2]

    # Align ref_top5 to the eval window: top5_tokens[i] is the prediction AFTER
    # seeing reference_tokens[0, i], so for eval positions [split_point, ..., n_total-2]
    # we need ref_top5[split_point : split_point + n_eval].
    assert ref_top5.shape[0] >= split_point + n_eval, (
        f"Reference top5 has only {ref_top5.shape[0]} rows; " f"need at least {split_point + n_eval}"
    )
    ref_top5_eval = ref_top5[split_point : split_point + n_eval]  # [n_eval, 5]

    mesh, wc = model_handles
    input_ids = reference_tokens[0].tolist()  # full sequence

    print(
        f"\n[accuracy] Teacher-forcing over {n_total} tokens " f"(eval on last {n_eval} positions)...",
        flush=True,
    )
    predicted_top1, predicted_top5 = run_teacher_forcing(mesh, wc, input_ids, n_eval)

    top1_acc, top5_acc = compute_accuracy(predicted_top1, predicted_top5, ref_top5_eval)
    print(f"[accuracy] top-1: {top1_acc:.3f}  top-5: {top5_acc:.3f}", flush=True)
    print(f"[accuracy] thresholds: top-1 >= {TOP1_THRESHOLD}  top-5 >= {TOP5_THRESHOLD}", flush=True)

    assert top5_acc >= TOP5_THRESHOLD, (
        f"top-5 accuracy {top5_acc:.3f} is below threshold {TOP5_THRESHOLD}. "
        "Check for DRAM corruption, SSM state bugs, or weight loading regressions."
    )
    assert top1_acc >= TOP1_THRESHOLD, (
        f"top-1 accuracy {top1_acc:.3f} is below threshold {TOP1_THRESHOLD}. "
        "Predictions diverged from reference — likely a numerical regression."
    )


@pytest.mark.slow
def test_token_accuracy_summary(model_handles):
    """Verbose variant: prints per-mismatch details for the first 20 failures.

    Useful for debugging when test_token_accuracy fails.
    """
    if not os.path.exists(REFERENCE_FILE):
        pytest.skip("No reference file — run test_generate_reference first.")

    ref = torch.load(REFERENCE_FILE, weights_only=False)
    reference_tokens = ref["reference_tokens"]  # [1, N]
    ref_top5 = ref["top5_tokens"]  # [N-1, 5]

    n_total = reference_tokens.shape[1]
    split_point = n_total // 2
    n_eval = n_total - split_point - 1
    ref_top5_eval = ref_top5[split_point : split_point + n_eval]

    mesh, wc = model_handles
    tokenizer = _load_tokenizer()
    input_ids = reference_tokens[0].tolist()

    predicted_top1, predicted_top5 = run_teacher_forcing(mesh, wc, input_ids, n_eval)

    ref_top1 = ref_top5_eval[:, 0]
    mismatches = (predicted_top1 != ref_top1).nonzero(as_tuple=True)[0]

    print(f"\n[summary] {len(mismatches)}/{n_eval} top-1 mismatches", flush=True)
    for idx in mismatches[:20]:
        pos = split_point + int(idx)
        ctx_ids = input_ids[max(0, pos - 5) : pos + 1]
        ctx = tokenizer.decode(ctx_ids)
        ours = tokenizer.decode([int(predicted_top1[idx])])
        ref = tokenizer.decode([int(ref_top1[idx])])
        print(f"  pos={pos:4d}  ctx='{ctx}'  ours='{ours}'  ref='{ref}'", flush=True)

    top1_acc, top5_acc = compute_accuracy(predicted_top1, predicted_top5, ref_top5_eval)
    print(f"[summary] top-1={top1_acc:.3f}  top-5={top5_acc:.3f}")
