# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-slot equivalence of narrowed prefill against the full-width path.

vLLM prefills one request per scheduler step, so a 32-slot server ran the whole
layer stack over 32 rows to fill one. Narrowing to the active row must be a pure
optimization: for every slot, the terminal logits and the per-slot conv,
recurrent and paged KV state have to match what the full-width path produced.

Slot 0 passing proves little -- a narrowed run that always wrote slot 0 would
look perfect there and silently corrupt every other user -- so this sweeps all
slots and checks the *untouched* slots are bit-identical as well.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.utility_functions import comp_pcc


def _min_pcc(lhs, rhs, bar):
    """Worst PCC over a list of paired state tensors."""
    worst, ok = 1.0, True
    assert len(lhs) == len(rhs), "state tensor count differs"
    for a, b in zip(lhs, rhs):
        if torch.equal(a, b):
            continue
        passed, value = comp_pcc(a.float(), b.float(), bar)
        try:
            value = float(str(value).split()[-1])
        except (ValueError, IndexError):
            value = 0.0 if not passed else 1.0
        worst = min(worst, value)
        ok = ok and passed
    return ok, worst


def _host(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def _linear_state(generator, slot):
    out = []
    for layer in generator.model.layers:
        if layer.layer_kind == "linear_attention":
            conv = _host(layer.caches["conv"])
            # Composite conv is [1, batch, channels, kernel]; the fused KDA path
            # stores the state as its row-major [batch, kernel, channels] window,
            # where the fixed slot is axis 0. Indexing axis 1 there would compare
            # kernel taps across slots and pass while measuring nothing.
            out.append((conv[:, slot] if conv.dim() == 4 else conv[slot]).clone())
            out.append(_host(layer.caches["recurrent"])[slot].clone())
    return out


def _kv_state(generator, slot):
    blocks = generator.page_table_host[slot].tolist()
    out = []
    for layer in generator.model.layers:
        if layer.layer_kind == "full_attention":
            for name in ("key", "value"):
                out.append(_host(layer.caches[name])[blocks].clone())
    return out


def _prefill_one_slot(generator, slot, batch, length, vocab):
    """Prefill a distinct prompt into ``slot`` only; all other slots inactive."""
    tokens = torch.zeros((batch, length), dtype=torch.long)
    # a slot-dependent prompt, so a mis-routed write cannot coincidentally match
    tokens[slot, :length] = (torch.arange(length) * 7 + slot * 101 + 3) % vocab
    prompt_lens = [0] * batch
    prompt_lens[slot] = length
    logits = generator.prefill_forward(
        tokens,
        page_table=generator._page_table,
        kv_cache=generator.kv_cache,
        prompt_lens=prompt_lens,
    )
    return logits[slot, 0].float().clone()


def run(mesh, *, batch, length, layers, max_context, slots, bar, state_bar):
    """One generator, toggled between paths -- same weights, same caches."""
    import os

    generator = build_generator(
        Path("models/autoports/qwen_qwen3_6_27b"),
        mesh,
        num_layers=layers,
        max_context=max_context,
        batch=batch,
    )
    vocab = generator.model.vocab_size
    failures = []
    for slot in slots:
        captured = {}
        for mode in ("0", "1"):
            os.environ["QWEN36_PREFILL_NARROW"] = mode
            generator.model.reset_cache()
            generator._slots_requiring_prefill = set(range(batch))
            logits = _prefill_one_slot(generator, slot, batch, length, vocab)
            captured[mode] = {
                "logits": logits,
                "linear": _linear_state(generator, slot),
                "kv": _kv_state(generator, slot),
                "other_linear": _linear_state(generator, (slot + 1) % batch),
            }
        wide, narrow = captured["0"], captured["1"]
        pcc_ok, pcc = comp_pcc(wide["logits"], narrow["logits"], bar)
        argmax_ok = int(wide["logits"].argmax()) == int(narrow["logits"].argmax())
        # Narrowing the batch changes matmul shapes, so accumulation order --
        # and therefore the last bits -- differ. Computed state is compared by
        # PCC. Exactness is demanded only of the slots this prefill must not
        # touch at all, which is the property that catches mis-routed writes.
        linear_pcc = _min_pcc(wide["linear"], narrow["linear"], state_bar)
        kv_pcc = _min_pcc(wide["kv"], narrow["kv"], state_bar)
        linear_ok, kv_ok = linear_pcc[0], kv_pcc[0]
        other_ok = all(torch.equal(a, b) for a, b in zip(wide["other_linear"], narrow["other_linear"]))
        print(
            f"SLOT {slot:2d} logits_pcc={pcc} argmax_match={argmax_ok} "
            f"linear_state_min_pcc={linear_pcc[1]:.8f} kv_state_min_pcc={kv_pcc[1]:.8f} "
            f"neighbour_bit_exact={other_ok}",
            flush=True,
        )
        if not (pcc_ok and argmax_ok and linear_ok and kv_ok and other_ok):
            failures.append(slot)
    os.environ.pop("QWEN36_PREFILL_NARROW", None)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--bar", type=float, default=0.999)
    # Recurrent state accumulates a per-step difference over the whole prompt,
    # so it diverges slightly more than the logits it produces. Measured range
    # over 8 slots at length 96: linear 0.99857-0.99904, kv 0.99943-0.99956,
    # logits 0.99966-0.99978 with every argmax identical.
    parser.add_argument("--state-bar", type=float, default=0.998)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    try:
        slots = list(range(args.batch))
        failures = run(
            mesh,
            batch=args.batch,
            length=args.length,
            layers=args.layers,
            max_context=args.max_context,
            slots=slots,
            bar=args.bar,
            state_bar=args.state_bar,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if args.output:
        args.output.write_text(json.dumps({"failing_slots": failures}, indent=2) + "\n")
    if failures:
        raise SystemExit(f"ACTIVE_ROW_PREFILL FAILED for slots {failures}")
    print("ACTIVE_ROW_PREFILL OK")


if __name__ == "__main__":
    main()
