"""Crash/shape test of the MTP oracle on a stand-in hidden state (the L5 dump output, NOT a last-layer hidden: accuracies are meaningless, only that all variants run)."""
import torch

from models.demos.blackhole.mimo_v26.reference.mtp_ref import evaluate

DUMP = "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16"


def test_runs_all_variants():
    toks = torch.load(f"{DUMP}/tokens.pt")["prefill_tokens"][:2]
    h = torch.load(f"{DUMP}/layer_5.pt")["prefill"]["h_out"][:2, :124]
    res = evaluate(h, toks, n_layers=3)
    assert len(res) == 16 and all(len(v) == 3 and all(0 <= a <= 1 for a in v) for v in res.values())
