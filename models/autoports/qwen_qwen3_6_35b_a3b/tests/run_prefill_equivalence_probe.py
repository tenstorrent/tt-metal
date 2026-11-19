# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight legacy/bounded prefill equivalence probe; run serialized.

Suggested bounded command after lane environment setup:
 timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_prefill_equivalence_probe

Use --section tails or --section full to rerun a failing surface directly.
This diagnostic uses layers 0 and 3, context 4096, and no traces/profiler. It
reports exactness separately from its numerical gate; --require-exact makes
any nonexact tensor comparison fail. It is not full-model acceptance evidence.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION, QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import _bounded_concat
from models.autoports.qwen_qwen3_6_35b_a3b.tt.prefill import BoundedPrefill

ROOT = Path(__file__).resolve().parents[1]


def tensor_metrics(reference, candidate, *, logits=False):
    if reference.shape != candidate.shape:
        return dict(passed=False, shape_reference=list(reference.shape), shape_candidate=list(candidate.shape))
    a, b = reference.float(), candidate.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    exact = bool(torch.equal(reference, candidate))
    if not finite:
        result = dict(shape=list(a.shape), finite=False, exact=exact, pcc=0.0, error="Nonfinite tensor")
        if logits:
            result.update(top1=0.0, top5=0.0, top100=0.0, greedy_token_agreement=0.0)
        return result
    delta = a - b
    # Chunked sufficient statistics avoid two full FP64 vocabulary tensors.
    af, bf = a.reshape(-1), b.reshape(-1)
    sa = sb = saa = sbb = sab = 0.0
    for offset in range(0, af.numel(), 1 << 20):
        aa, bb = af[offset : offset + (1 << 20)].double(), bf[offset : offset + (1 << 20)].double()
        sa += aa.sum().item()
        sb += bb.sum().item()
        saa += aa.square().sum().item()
        sbb += bb.square().sum().item()
        sab += (aa * bb).sum().item()
    count = af.numel()
    va, vb = max(0.0, saa - sa * sa / count), max(0.0, sbb - sb * sb / count)
    pcc = 1.0 if exact else ((sab - sa * sb / count) / (va * vb) ** 0.5 if va and vb else 0.0)
    result = dict(
        shape=list(a.shape),
        finite=finite,
        exact=exact,
        pcc=pcc,
        max_abs=float(delta.abs().max()),
        rms=float(delta.square().mean().sqrt()),
        unequal_elements=int(torch.count_nonzero(delta)),
        elements=count,
    )
    if logits:
        width = a.shape[-1]
        predictions = b.argmax(-1)
        ranked = torch.topk(a, min(100, width), dim=-1).indices
        for k in (1, 5, 100):
            result[f"top{k}"] = float((ranked[..., : min(k, width)] == predictions[..., None]).any(-1).float().mean())
        result["greedy_token_agreement"] = float((a.argmax(-1) == predictions).float().mean())
        result["first_greedy_mismatch"] = (a.argmax(-1) != predictions).reshape(-1).nonzero().flatten()[:1].tolist()
    return result


def local_hosts(tensor):
    return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(tensor)]


def compare_chips(reference, candidate):
    assert len(reference) == len(candidate) == 4
    return [tensor_metrics(a, b) for a, b in zip(reference, candidate)]


class ObservedPrefill(BoundedPrefill):
    def __init__(self, model):
        super().__init__(model)
        self.observe = False
        self.observed = {}
        self.valid_length = 0

    def _moe(self, layer, x):
        output = super()._moe(layer, x)
        if self.observe:
            prior = sum(item[0].shape[1] for item in self.observed.get(id(layer), []))
            valid = min(x.shape[1], self.valid_length - prior)
            self.observed.setdefault(id(layer), []).append([host[:, :valid] for host in local_hosts(output)])
        return output

    def begin_observation(self, length):
        self.valid_length = length
        self.observed = {}
        self.observe = True

    def layer_outputs(self, layer):
        chunks = self.observed[id(layer)]
        return [torch.cat([chunk[chip] for chunk in chunks], dim=1) for chip in range(4)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--section", choices=("all", "tails", "full"), default="all")
    p.add_argument("--tail-lengths", default="2,31,33,65,2047")
    p.add_argument("--prompt-lengths", default="1,31,33,65,257")
    p.add_argument("--continuations", default="31:2,31:33,255:2,255:33")
    p.add_argument("--require-exact", action="store_true")
    p.add_argument("--token-fixture", type=Path, help="Exact CPU token IDs for a causal comparison")
    p.add_argument(
        "--drain-legacy-attention",
        action="store_true",
        help="Store legacy attention results in DRAM immediately; numerical reference only",
    )
    p.add_argument("--pcc-min", type=float, default=0.99999)
    p.add_argument("--output", type=Path, default=ROOT / "doc/full_model/prefill_equivalence/result.json")
    args = p.parse_args()
    tails = [int(n) for n in args.tail_lengths.split(",") if n]
    lengths = [int(n) for n in args.prompt_lengths.split(",") if n]
    continuations = [tuple(map(int, item.split(":"))) for item in args.continuations.split(",") if item]
    assert all(1 <= n <= 2048 for n in tails)
    assert all(0 <= start < start + n <= 4096 for start, n in [(0, n) for n in lengths] + continuations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        *[
            ROOT / "tt" / n
            for n in ("model.py", "multichip_decoder.py", "prefill.py", "prefill_transport.py", "slot_state.py")
        ],
    ]
    result = dict(
        status="RUNNING",
        model=HF_MODEL,
        revision=HF_REVISION,
        layers=[0, 3],
        context=4096,
        command=sys.argv,
        timestamp=time.time(),
        section=args.section,
        legacy_attention_dram_drain=args.drain_legacy_attention,
        gate=dict(
            pcc_min=args.pcc_min,
            top5_min=0.98,
            top100=1.0,
            require_exact=args.require_exact,
            scope="diagnostic equivalence, not full-model acceptance",
        ),
        source_sha256={str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={
            name: os.environ.get(name)
            for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER")
        },
        tails=[],
        prefill=[],
        failures=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def classify(metric, label):
        passed = metric.get("finite", False) and metric.get("pcc", -1) >= args.pcc_min
        if "top5" in metric:
            passed = passed and metric["top5"] >= 0.98 and metric["top100"] == 1.0
        if args.require_exact:
            passed = passed and metric.get("exact", False)
        metric["passed"] = bool(passed)
        if not passed:
            result["failures"].append(label)
        return metric

    def checked_chips(reference, candidate, label):
        return [classify(m, f"{label}/chip{chip}") for chip, m in enumerate(compare_chips(reference, candidate))]

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    save()
    try:
        model = QwenModel(mesh, layer_indices=[0, 3], max_seq_len=4096)
        caches = [model.allocate_cache(context_length=4096), model.allocate_cache(context_length=4096)]
        zeros = {}
        for cache in caches:
            for states in cache:
                for key, value in states[0].items():
                    if key in ("k", "v", "conv", "recurrent"):
                        zeros[id(value)] = ttnn.zeros_like(value)

        def reset(cache):
            for states in cache:
                for key, value in states[0].items():
                    if key in ("k", "v", "conv", "recurrent"):
                        ttnn.copy(zeros[id(value)], value)

        def snapshot(cache):
            return {
                f"layer{index}/{key}": local_hosts(value)
                for index, states in zip(model.layer_indices, cache)
                for key, value in states[0].items()
                if key in ("k", "v", "conv", "recurrent")
            }

        required = max([1, *tails, *lengths, *[s + n for s, n in continuations]])
        if args.token_fixture is not None:
            ids = json.loads(args.token_fixture.read_text())
            assert isinstance(ids, list) and all(isinstance(x, int) for x in ids)
            result["token_fixture"] = dict(
                source=str(args.token_fixture),
                mode="exact numerical-control token fixture; no qualitative judgment",
                token_ids=ids[:required],
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model.checkpoint.path, local_files_only=True)
            fixture_text = "Explain how supervised and unsupervised learning differ, with concrete examples. " * 400
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": fixture_text}], tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            result["token_fixture"] = dict(
                tokenizer=type(tokenizer).__name__,
                chat_template=True,
                mode="truncated chat-template activation fixture; not a qualitative judgment",
                token_ids=ids[:required],
            )
        assert len(ids) >= required
        bounded = ObservedPrefill(model)

        if args.section in ("all", "tails"):
            for n in tails:
                reset(caches[0])
                tokens = model.upload(
                    torch.tensor([ids[:n]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                hidden = ttnn.reshape(model.embed(tokens), (1, n, model.config.hidden_size))
                for index, layer, states in zip(model.layer_indices, model.layers, caches[0]):
                    state = states[0]
                    plan = layer.prepare_prefill(seq_len=n, start_pos=0)
                    BoundedPrefill._begin_layer(layer, state)
                    attention = []
                    for offset, rows, absolute, positions in plan:
                        attention.append(
                            layer._attention_residual(
                                hidden[:, offset : offset + rows, :],
                                state=state,
                                decode=rows == 1 and (layer.kind == "full_attention" or n == 1),
                                positions=positions,
                                start_pos=absolute,
                                cache_boundary=False,
                                projection_k=1024 if n > 1 else 512,
                            )
                        )
                    attended = (
                        _bounded_concat(
                            [ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG) for x in attention], dim=1, fan_in=64
                        )
                        if len(attention) > 1
                        else attention[0]
                    )
                    # Exact original tail: normalize n logical rows, then let
                    # the selected decoder own its internal MoE padding.
                    norm = layer._normalized_input(attended, "post_attention_layernorm.weight")
                    reduced = layer._reduce(layer._moe(norm))
                    reference = ttnn.clone(ttnn.add(attended, reduced), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    # Snapshot each boundary before invoking a second MoE:
                    # shared dispatch/CCL workspaces must not hide differences.
                    reference_norm_host = local_hosts(norm)
                    reference_moe_host = local_hosts(reduced)
                    reference_hidden_host = local_hosts(reference)
                    physical = 1 if n == 1 else (n + 31) // 32 * 32
                    bounded._write(attended, bounded.residual[1], 0, n)
                    padded = bounded._read(bounded.residual[1], 0, physical, valid_rows=n)
                    padded_norm = layer._normalized_input(padded, "post_attention_layernorm.weight")
                    padded_moe = layer._reduce(layer._moe(padded_norm))
                    candidate_norm_host = [h[:, :n] for h in local_hosts(padded_norm)]
                    candidate_moe_host = [h[:, :n] for h in local_hosts(padded_moe)]
                    candidate = BoundedPrefill._moe(layer, padded)
                    prefix = f"tail/n{n}/layer{index}"
                    entry = dict(
                        length=n,
                        physical_rows=physical,
                        layer=index,
                        kind=layer.kind,
                        norm=checked_chips(reference_norm_host, candidate_norm_host, prefix + "/norm"),
                        moe=checked_chips(reference_moe_host, candidate_moe_host, prefix + "/moe"),
                        residual=checked_chips(
                            reference_hidden_host, [h[:, :n] for h in local_hosts(candidate)], prefix + "/residual"
                        ),
                    )
                    # Last-row LM head gives meaningful token top-k without
                    # materializing 2047 x full-vocabulary logits in this probe.
                    ref_logits = model.host_logits(model.terminal(reference[:, -1:, :]))
                    cand_logits = model.host_logits(model.terminal(candidate[:, n - 1 : n, :]))
                    entry["last_logits"] = classify(
                        tensor_metrics(ref_logits, cand_logits, logits=True), prefix + "/last_logits"
                    )
                    result["tails"].append(entry)
                    hidden = reference
                    print(
                        f"TAIL_EQUIVALENCE n={n} layer={index} exact={entry['residual'][0]['exact']} pcc={entry['residual'][0]['pcc']}",
                        flush=True,
                    )
                    save()
                    del attention, attended, norm, reduced, padded, padded_norm, padded_moe, candidate
                del hidden, tokens
                gc.collect()

        if args.section in ("all", "full"):

            def restore_all():
                for cache in caches:
                    reset(cache)

            bounded.prepare(
                cache=caches[1],
                reset_state=restore_all,
                progress=lambda row: print("BOUNDED_PREPARE " + json.dumps(row), flush=True),
            )
            result["bounded_prepare"] = dict(
                completed=True,
                workspace_bytes_per_chip=bounded.workspace_bytes_per_chip,
                program_cache_entries=int(mesh.num_program_cache_entries()),
            )
            legacy_observed = {}
            originals = [layer.prefill_forward for layer in model.layers]
            for layer, original in zip(model.layers, originals):

                def observed(hidden_states, *, state, plan, layer=layer, original=original):
                    output = original(hidden_states, state=state, plan=plan)
                    legacy_observed[id(layer)] = local_hosts(output)
                    return output

                layer.prefill_forward = observed

            def run_legacy(tokens, start):
                tt_tokens = model.upload(
                    torch.tensor([tokens], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                attention_originals = [layer._attention_residual for layer in model.layers]
                try:
                    if args.drain_legacy_attention:
                        for layer, original in zip(model.layers, attention_originals):

                            def drained(*values, original=original, **kwargs):
                                return ttnn.to_memory_config(original(*values, **kwargs), ttnn.DRAM_MEMORY_CONFIG)

                            layer._attention_residual = drained
                    out = model.prefill(tt_tokens, cache=caches[0], start=start, return_all_logits=True)
                    return model.host_logits(out)
                finally:
                    for layer, original in zip(model.layers, attention_originals):
                        layer._attention_residual = original

            def run_bounded(tokens, start):
                chunks = []

                def collect(offset, valid, logits):
                    chunks.append((offset, model.host_logits(logits)[:, :valid, :]))

                last = bounded.prefill(tokens, cache=caches[1], start=start, on_logits=collect)
                expected_offsets = list(range(0, len(tokens), bounded.LOGIT_CHUNK))
                assert [offset for offset, _ in chunks] == expected_offsets
                logits = torch.cat([chunk for _, chunk in chunks], dim=1)
                last_metric = tensor_metrics(logits[:, -1:, :], model.host_logits(last), logits=True)
                return logits, last_metric

            try:
                for start, n in [(0, n) for n in lengths] + continuations:
                    restore_all()
                    bounded.observe = False
                    if start:
                        run_legacy(ids[:start], 0)
                        run_bounded(ids[:start], 0)
                    legacy_observed.clear()
                    bounded.begin_observation(n)
                    reference = run_legacy(ids[start : start + n], start)
                    ref_states = snapshot(caches[0])
                    candidate, last_metric = run_bounded(ids[start : start + n], start)
                    candidate_states = snapshot(caches[1])
                    prefix = f"prefill/start{start}/n{n}"
                    entry = dict(
                        start=start,
                        length=n,
                        logits=classify(tensor_metrics(reference, candidate, logits=True), prefix + "/logits"),
                        bounded_last_vs_callback=classify(last_metric, prefix + "/last_callback"),
                        layers={
                            str(index): checked_chips(
                                legacy_observed[id(layer)], bounded.layer_outputs(layer), prefix + f"/layer{index}"
                            )
                            for index, layer in zip(model.layer_indices, model.layers)
                        },
                        cache={
                            key: checked_chips(value, candidate_states[key], prefix + "/cache/" + key)
                            for key, value in ref_states.items()
                        },
                    )
                    result["prefill"].append(entry)
                    bounded.observe = False
                    print(
                        f"PREFILL_EQUIVALENCE start={start} n={n} exact={entry['logits']['exact']} top5={entry['logits']['top5']} top100={entry['logits']['top100']}",
                        flush=True,
                    )
                    save()
                    del reference, candidate, ref_states, candidate_states
                    gc.collect()
            finally:
                for layer, original in zip(model.layers, originals):
                    layer.prefill_forward = original
        result["status"] = "PASS" if not result["failures"] else "FAIL"
        result["all_compared_tensors_exact"] = all(item.get("exact", True) for item in _metric_nodes(result))
        save()
        assert not result["failures"], result["failures"]
        print(f"PREFILL_EQUIVALENCE_PASS {args.output}", flush=True)
    except BaseException:
        result["status"] = "ERROR" if result["status"] == "RUNNING" else result["status"]
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


def _metric_nodes(value):
    if isinstance(value, dict):
        if "exact" in value and "pcc" in value:
            yield value
        for child in value.values():
            yield from _metric_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _metric_nodes(child)


if __name__ == "__main__":
    main()
