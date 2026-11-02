"""QWEN38_DR2_COMPARE=1 (device unit test, tests/test_draft_row_1row_device.py): the chat session's eager MTP warm rounds run
the verify-form draft rows and the 1-row draft rows on the same state and print one ``DR2 {json}`` line per round.

QWEN38_DR2_POSITIONS (comma list, increasing, >= 6 apart) sets the positions P of the rounds."""

import json
import os

import torch

import ttnn

KV_DEVICES = (0, 7)


def positions() -> list[int]:
    return [int(x) for x in os.environ.get("QWEN38_DR2_POSITIONS", "").split(",") if x.strip()]


def _hidden_host(hidden) -> torch.Tensor:
    return torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(hidden)], dim=-1)


def _snapshot(alignment, position: int) -> dict:
    state = alignment.generic_state.attention
    devices = ttnn.get_device_tensors(state.packed_kv_cache)
    blocks = ttnn.get_device_tensors(state.compressed_index_cache)
    kv = {d: ttnn.to_torch(devices[d])[..., :position, :].clone() for d in KV_DEVICES}
    comp = {d: ttnn.to_torch(blocks[d])[..., : position // 4, :].clone() for d in KV_DEVICES}
    return {"kv": kv, "comp": comp}


def _same(a: dict, b: dict) -> bool:
    return all(
        torch.equal(a[k][d].view(torch.int16), b[k][d].view(torch.int16)) for k in ("kv", "comp") for d in KV_DEVICES
    )


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def run(mtp_v2, model, target, state, output, position: int, synchronize) -> None:
    verify, draft = target.verify, target.draft
    alignment = verify.alignment
    rows = {"old": {}, "new": {}}

    def probe(name):
        def call(row, hidden):
            rows[name][row] = _hidden_host(hidden)

        return call

    before = _snapshot(alignment, position)
    mtp_v2.forward_draft(model, verify, draft, state, output, hidden_probe=probe("old"))
    synchronize()
    _, tokens_old = mtp_v2.read_pass_row(verify, draft)
    after_old = _snapshot(alignment, position)
    mtp_v2.forward_draft_1row(model, verify, draft, state, output, hidden_probe=probe("new"))
    synchronize()
    _, tokens_new = mtp_v2.read_pass_row(verify, draft)
    after_new = _snapshot(alignment, position)
    per_row = []
    for row in sorted(rows["old"]):
        a, b = rows["old"][row], rows["new"][row]
        per_row.append(
            {
                "row": row,
                "pcc": _pcc(a, b),
                "max_abs": float((a - b).abs().max()),
                "ref_max_abs": float(a.abs().max()),
                "token_old": tokens_old[2 + row],
                "token_new": tokens_new[2 + row],
            }
        )
    print(
        "DR2 "
        + json.dumps(
            {
                "position": position,
                "p_mod_4": position % 4,
                "p_mod_32": position % 32,
                "tokens_old": list(tokens_old),
                "tokens_new": list(tokens_new),
                "rows": per_row,
                "committed_unchanged_old": _same(before, after_old),
                "committed_unchanged_new": _same(before, after_new),
            }
        ),
        flush=True,
    )
