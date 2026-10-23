# SPDX-License-Identifier: Apache-2.0
"""Optimized-path HF correctness, trace safety and profiling. Run as a module."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb,
)

from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder, LinearState, PagedState
import os


def pcc(a, b):
    # Large real-shape outputs need FP64 statistics: FP32 correlation can
    # round/clamp to 1.0 when accumulating tens of millions of elements.
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


class NoTorchRuntime(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f'Host torch operation inside decoder pass: {func}')


def audit_runtime(function):
    def checked(*args, **kwargs):
        with NoTorchRuntime(), patch.object(ttnn, 'from_torch', side_effect=AssertionError('runtime from_torch')), \
                patch.object(ttnn, 'to_torch', side_effect=AssertionError('runtime to_torch')):
            return function(*args, **kwargs)
    return checked


@torch.no_grad()
def reference_prefill(ref, x, cos, sin, cache):
    """Long tests compare the final 128 outputs against all valid prefix state.

    Earlier tokens need only the token mixer to establish this single layer's
    state. This avoids quadratic reference outputs without shortening context.
    TT still computes and returns every output.
    """
    batch, length, _ = x.shape
    tail = min(length, 128) if batch * length > 4096 else length
    prefix = length - tail
    if prefix and ref.layer_type == 'full_attention':
        ks, vs = [], []
        for start in range(0, prefix, 2048):
            end = min(start + 2048, prefix)
            h = ref.input_layernorm(x[:, start:end])
            a = ref.self_attn
            k = a.k_norm(a.k_proj(h).reshape(batch, end-start, 4, 256)).transpose(1, 2)
            v = a.v_proj(h).reshape(batch, end-start, 4, 256).transpose(1, 2)
            _, k = apply_rotary_pos_emb(k, k, cos[:, start:end], sin[:, start:end])
            ks.append(k); vs.append(v)
        cache.update(torch.cat(ks, dim=2), torch.cat(vs, dim=2), ref.self_attn.layer_idx)
    elif prefix:
        for start in range(0, prefix, 1024):
            end = min(start + 1024, prefix)
            ref.linear_attn(ref.input_layernorm(x[:, start:end]), cache_params=cache)
    mask = torch.where(torch.arange(length)[None, :] > torch.arange(prefix, length)[:, None],
                       float('-inf'), 0.0).to(torch.bfloat16)[None, None]
    return ref(x[:, prefix:], position_embeddings=(cos[:, prefix:], sin[:, prefix:]),
               attention_mask=mask if ref.layer_type == 'full_attention' else None, past_key_values=cache)


def main():
    autoport = Path(__file__).resolve().parents[1]
    provenance = {str(p.relative_to(autoport)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__), *sorted((autoport / 'tt').rglob('*.py')),
                            *sorted((autoport / 'tt').rglob('*.cpp'))]}
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-functional', action='store_true')
    parser.add_argument('--workload', action='store_true', help='4096+128 workload without profiler; host latency only')
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--seq', type=int, default=32)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prefill-only', action='store_true')
    parser.add_argument('--trace-prefill', action='store_true', help='Capture warmed fixed-shape prefill and measure its replay')
    parser.add_argument('--profile', action='store_true', help='4096 prefill + 128 traced teacher-input decode steps')
    parser.add_argument('--decode-profile-only', action='store_true', help='Profile decode only after the normal4096 prefill')
    parser.add_argument('--debug-states', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--recorded', action='store_true')
    parser.add_argument('--baseline-fused', action='store_true')
    parser.add_argument('--inject-capture-failure', action='store_true')
    parser.add_argument('--remap', action='store_true', help='change page ownership under the same live decode trace')
    parser.add_argument('--requests', action='store_true', help='increasing and short-reuse requests on one loaded decoder')
    parser.add_argument('--input-std', type=float, default=0.1)
    parser.add_argument('--weights', default='.cache/qwen38-weights')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    global LinearState, PagedState
    decoder_class = OptimizedDecoder
    if args.baseline_fused:
        from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder, LinearState, PagedState
        decoder_class = FusedDecoder
    assert not args.decode_profile_only or args.profile
    if args.workload:
        assert args.seq == 4096
    if args.profile:
        assert args.seq == 4096
        assert not args.remap, 'Profile and page-remapping validation use separate runs'
        from tracy import signpost
    torch.set_num_threads(8)
    torch.manual_seed(982)
    evidence = Path(__file__).resolve().parents[1] / 'doc' / 'functional_decoder'
    cfg = AutoConfig.from_pretrained(str(evidence / 'hf_config.json')).text_config
    cfg._attn_implementation = 'eager'
    prefix = f'model.language_model.layers.{args.layer}.'
    if args.synthetic:
        generator = torch.Generator().manual_seed(417)
        weights = {
            row['name']: (torch.randn(row['shape'], generator=generator) * row['std'] + row['mean']).to(torch.bfloat16)
            for row in json.loads((evidence / 'weight_stats.json').read_text()) if row['name'].startswith(prefix)
        }
    else:
        weights = load_file(str(Path(args.weights) / f'layer_{args.layer}.safetensors'))
    layer_weights = {k.removeprefix(prefix): v for k, v in weights.items()}
    ref = Qwen3_5DecoderLayer(cfg, args.layer).to(torch.bfloat16).eval()
    ref.load_state_dict(layer_weights, strict=True)
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    cache = DynamicCache(config=cfg)
    x = torch.randn(args.batch, args.seq, cfg.hidden_size, dtype=torch.bfloat16) * args.input_std
    xd = torch.randn(args.batch, 1, cfg.hidden_size, dtype=torch.bfloat16) * args.input_std
    xd2 = torch.randn_like(xd) * args.input_std
    if args.recorded:
        recorded = torch.load(Path(args.weights) / f'recorded_layer_{args.layer}.pt', weights_only=True)
        def recorded_inputs(length, start=0):
            # Distinct real activation windows per user; repeat the observed
            # fixture for long single-layer capacity tests, without truncation.
            idx = (torch.arange(start,start+length)[None,:] + 17*torch.arange(args.batch)[:,None]) % recorded.shape[1]
            return recorded[0,idx]
        x = recorded_inputs(args.seq)
        xd = recorded_inputs(1,args.seq)
        xd2 = recorded_inputs(1,args.seq+1)
    positions = torch.arange(args.seq).expand(3, args.batch, -1)
    cos, sin = rope(x, positions)
    cd, sd = rope(xd, torch.full((3, args.batch, 1), args.seq, dtype=torch.long))
    cd2, sd2 = rope(xd2, torch.full((3, args.batch, 1), args.seq + 1, dtype=torch.long))
    with torch.no_grad():
        expected = reference_prefill(ref, x, cos, sin, cache)
        if args.debug_states and ref.layer_type == 'linear_attention':
            expected_state = cache.layers[args.layer].recurrent_states.clone()
            expected_conv = cache.layers[args.layer].conv_states[:, :, -3:].transpose(1, 2).clone()
        workload_reference = None
        if args.profile or args.workload:
            import copy
            workload_cache = copy.deepcopy(cache)
            for step in range(128):
                wc, ws = rope(xd, torch.full((3, args.batch, 1), args.seq + step, dtype=torch.long))
                workload_reference = ref(xd, position_embeddings=(wc, ws), past_key_values=workload_cache)
            del workload_cache
        if not args.prefill_only:
            expected_decode = ref(xd, position_embeddings=(cd, sd), past_key_values=cache)
            if args.remap and ref.layer_type == 'full_attention':
                expected_key = cache.layers[args.layer].keys[:, :, args.seq:args.seq+1].clone()
                expected_value = cache.layers[args.layer].values[:, :, args.seq:args.seq+1].clone()
            expected_decode2 = ref(xd2, position_embeddings=(cd2, sd2), past_key_values=cache)
    print('HF_REFERENCE_READY', flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    tid = None
    capturing = False
    try:
        def dram_view():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return dict(num_banks=view.num_banks, capacity_bytes=view.total_bytes_per_bank * view.num_banks,
                        allocated_bytes=view.total_bytes_allocated_per_bank * view.num_banks,
                        largest_free_per_bank=view.largest_contiguous_bytes_free_per_bank)

        memory = {'before_weights': dram_view()}
        options = {} if args.baseline_fused else {'policy':json.loads(os.environ.get("QWEN_OPT_POLICY", "{}"))}
        decoder = decoder_class.from_state_dict(weights, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, **options)
        memory['after_weights'] = dram_view()
        if not args.profile:
            decoder.prefill_forward = audit_runtime(decoder.prefill_forward)
            decoder.decode_forward = audit_runtime(decoder.decode_forward)

        def put(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        capacity = min(cfg.max_position_embeddings, ((args.seq + 128 + 255) // 256) * 256)
        request_results = []
        if args.requests:
            selected = getattr(decoder, "policy", {})
            chunk = selected.get("prefill_chunk", decoder.chunk_size)
            norm_boundary = selected.get("prefill_norm_max_tokens", 512) // args.batch
            lengths = list(dict.fromkeys([1, 2, 3, 4, 31, 32, 33, 127, 128, 129, 385, 511, 512, 513,
                                         chunk-1, chunk, chunk+1, max(1, norm_boundary), norm_boundary+1]))
            for request_length in [*lengths, 31]:
                rx = torch.randn(args.batch, request_length, cfg.hidden_size, dtype=torch.bfloat16) * args.input_std
                rd = torch.randn(args.batch, 1, cfg.hidden_size, dtype=torch.bfloat16) * args.input_std
                if args.recorded:
                    rx, rd = recorded_inputs(request_length), recorded_inputs(1,request_length)
                rc, rs = rope(rx, torch.arange(request_length).expand(3, args.batch, -1))
                rdc, rds = rope(rd, torch.full((3, args.batch, 1), request_length, dtype=torch.long))
                reference_cache = DynamicCache(config=cfg)
                with torch.no_grad():
                    ry = reference_prefill(ref, rx, rc, rs, reference_cache)
                    rdy = ref(rd, position_embeddings=(rdc, rds), past_key_values=reference_cache)
                request_capacity = ((request_length + 1 + 255) // 256) * 256
                request_pages = request_capacity // 32
                assert request_length < request_capacity
                request_state = decoder.allocate_state(batch_size=args.batch, context_length=request_capacity)
                rpt = put(torch.randperm(args.batch * request_pages, dtype=torch.int32).reshape(args.batch, request_pages),
                          ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                rout = decoder.prefill_forward(put(rx), state=request_state, page_table=rpt,
                                               cos=put(rc[:, None]), sin=put(rs[:, None]))
                rpcc = pcc(ttnn.to_torch(rout)[:, -ry.shape[1]:], ry)
                rdout = decoder.decode_forward(put(rd), state=request_state, page_table=rpt,
                    current_pos=put(torch.full((args.batch,), request_length, dtype=torch.int32),
                                    ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), cos=put(rdc[:, None]), sin=put(rds[:, None]))
                rdpcc = pcc(ttnn.to_torch(rdout), rdy)
                request_results.append(dict(length=request_length, capacity=request_capacity, prefill_pcc=rpcc, decode_pcc=rdpcc))
                print('REQUEST', request_results[-1], flush=True)
                assert rpcc >= .995 and rdpcc >= .995
        state = decoder.allocate_state(batch_size=args.batch, context_length=capacity)
        empty_state = decoder.snapshot(state) if args.profile or args.timing or args.trace_prefill else None
        xt, xdt = put(x), put(xd)
        ct, st = put(cos[:, None]), put(sin[:, None])
        cdt, sdt = put(cd[:, None]), put(sd[:, None])
        pt_host = torch.randperm(args.batch * (capacity // 32), dtype=torch.int32).reshape(args.batch, -1)
        pt = put(pt_host, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = put(torch.full((args.batch,), args.seq, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        memory['request_setup'] = dram_view()
        actual = decoder.prefill_forward(xt, state=state, page_table=pt, cos=ct, sin=st)
        actual_host = ttnn.to_torch(actual)
        prefill_pcc = pcc(actual_host[:, -expected.shape[1]:], expected)
        compared = actual_host[:, -expected.shape[1]:].float().reshape(-1)
        wanted = expected.float().reshape(-1)
        prefill_pcc_fp32 = torch.corrcoef(torch.stack([compared, wanted]))[0, 1].item()
        prefill_relative_rmse = ((compared - wanted).square().mean() / wanted.square().mean()).sqrt().item()
        print('PREFILL_PCC', prefill_pcc, flush=True)
        fusion_prefill_pcc = fusion_decode_pcc = None
        control_decode = None
        if args.compare_functional:
            from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder
            control = FunctionalDecoder.from_state_dict(weights, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)
            control_state = control.allocate_state(batch_size=args.batch, context_length=capacity)
            control_prefill = control.prefill_forward(xt, state=control_state, page_table=pt, cos=ct, sin=st)
            fusion_prefill_pcc = pcc(actual_host, ttnn.to_torch(control_prefill))
            control_decode = ttnn.to_torch(control.decode_forward(xdt, state=control_state,
                page_table=pt, current_pos=pos, cos=cdt, sin=sdt)).clone()
            assert fusion_prefill_pcc >= .995
            print('FUSED_VS_FUNCTIONAL_PREFILL_PCC', fusion_prefill_pcc, flush=True)

        prefill_host_us = None
        prefill_deterministic = None
        if args.timing and not args.profile:
            for field in (('conv', 'recurrent') if isinstance(state, LinearState) else ('key', 'value')):
                ttnn.copy(getattr(empty_state, field), getattr(state, field))
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            warm_prefill = decoder.prefill_forward(xt, state=state, page_table=pt, cos=ct, sin=st)
            ttnn.synchronize_device(mesh)
            prefill_host_us = (time.perf_counter()-started)*1e6
            warm_host = ttnn.to_torch(warm_prefill)
            assert pcc(warm_host[:, -expected.shape[1]:], expected) >= .995
            prefill_deterministic = torch.equal(warm_host, actual_host)
            assert prefill_deterministic
            del warm_host
            del warm_prefill
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)  # drain outside either measured window
        if args.debug_states and ref.layer_type == 'linear_attention':
            for label, actual_state, wanted in [('recurrence', state.recurrent, expected_state),
                                                ('convolution', state.conv, expected_conv)]:
                got = ttnn.to_torch(actual_state).float()
                print('STATE', label, 'PCC', pcc(got, wanted), 'MAXERR', (got-wanted).abs().max().item(), flush=True)
        if args.prefill_only:
            Path(args.output).write_text(json.dumps(dict(layer_type=ref.layer_type, batch=args.batch,
                sequence=args.seq, prefill_pcc=prefill_pcc, compared_tokens=expected.shape[1],
                weights='synthetic' if args.synthetic else 'real', provenance=provenance,
                runtime_class=type(decoder).__name__, policy=getattr(decoder, 'policy', {}),
                activations='recorded HF preceding layers (repeated for long context)' if args.recorded else 'synthetic',
                runtime_fallback_audit=not args.profile, memory=memory), indent=2)+'\n')
            assert prefill_pcc >= .995
            return
        resume_pcc = None
        if args.resume:
            assert args.seq > 128
            other = decoder.allocate_state(batch_size=args.batch, context_length=capacity)
            first = decoder.prefill_forward(put(x[:, :128]), state=other, page_table=pt,
                                            cos=put(cos[:, None, :128]), sin=put(sin[:, None, :128]))
            checkpoint = decoder.snapshot(other)
            resumed = decoder.prefill_forward(put(x[:, 128:]), state=checkpoint, start_position=128, page_table=pt,
                                              cos=put(cos[:, None, 128:]), sin=put(sin[:, None, 128:]))
            resume_pcc = pcc(ttnn.to_torch(resumed), actual_host[:, 128:])
            print('RESUME_PCC', resume_pcc, flush=True)
        backup = decoder.snapshot(state)
        remapped = None
        if args.remap and ref.layer_type == 'full_attention':
            new_pt = pt_host.flip(1)
            moved = []
            for tensor in (backup.key, backup.value):
                old = ttnn.to_torch(tensor)
                new = torch.empty_like(old)
                new[new_pt.flatten().long()] = old[pt_host.flatten().long()]
                moved.append(put(new, dtype=tensor.dtype))
            remapped = PagedState(*moved)
            remap_refresh = [
                (ttnn.from_torch(new_pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), pt),
                (ttnn.from_torch(xd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), xdt),
                (ttnn.from_torch(cd[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cdt),
                (ttnn.from_torch(sd[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sdt),
                (ttnn.from_torch(torch.full((args.batch,), args.seq, dtype=torch.int32),
                                 dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), pos),
            ]

        def restore():
            fields = ('conv', 'recurrent') if isinstance(state, LinearState) else ('key', 'value')
            for field in fields:
                ttnn.copy(getattr(backup, field), getattr(state, field))

        if (args.profile and not args.decode_profile_only) or args.trace_prefill:
            def clear_prefill_state():
                for field in (('conv', 'recurrent') if isinstance(state, LinearState) else ('key', 'value')):
                    ttnn.copy(getattr(empty_state, field), getattr(state, field))
            clear_prefill_state()
            if args.trace_prefill:
                warm_prefill = decoder.prefill_forward(xt, state=state, page_table=pt, cos=ct, sin=st)
                del warm_prefill
                clear_prefill_state()
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                capturing = True
                measured_prefill = decoder.prefill_forward(xt, state=state, page_table=pt, cos=ct, sin=st)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                capturing = False
                clear_prefill_state()
            ttnn.synchronize_device(mesh)
            if args.profile:
                signpost('PERF_PREFILL')
            started = time.perf_counter()
            if args.trace_prefill:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            else:
                measured_prefill = decoder.prefill_forward(xt, state=state, page_table=pt, cos=ct, sin=st)
            ttnn.synchronize_device(mesh)
            prefill_host_us = (time.perf_counter()-started)*1e6
            if args.profile:
                signpost('PERF_PREFILL_END')
                ttnn.ReadDeviceProfiler(mesh)
            measured_host = ttnn.to_torch(measured_prefill)
            assert pcc(measured_host[:, -expected.shape[1]:], expected) >= .995
            prefill_deterministic = torch.equal(measured_host, actual_host)
            assert prefill_deterministic
            del measured_host
            if args.trace_prefill:
                ttnn.release_trace(mesh, tid)
                tid = None
            del measured_prefill
            restore()

        def decode():
            return decoder.decode_forward(xdt, state=state, page_table=pt, current_pos=pos, cos=cdt, sin=sdt)

        # Host-only refresh tensors are prepared before capture; no new device allocation on replay.
        refresh = [
            (ttnn.from_torch(xd2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), xdt),
            (ttnn.from_torch(cd2[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cdt),
            (ttnn.from_torch(sd2[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sdt),
            (ttnn.from_torch(torch.full((args.batch,), args.seq + 1, dtype=torch.int32),
                             dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), pos),
        ]
        perf_refresh = []
        if args.profile or args.workload:
            # Single-layer workload uses supplied hidden activations, not token generation.
            # All 128 steps run; only the final warmed complete layer is signposted.
            for step in range(128):
                cstep, sstep = rope(xd, torch.full((3, args.batch, 1), args.seq + step, dtype=torch.long))
                perf_refresh.append([
                    (ttnn.from_torch(xd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), xdt),
                    (ttnn.from_torch(cstep[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cdt),
                    (ttnn.from_torch(sstep[:, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sdt),
                    (ttnn.from_torch(torch.full((args.batch,), args.seq + step, dtype=torch.int32),
                                     dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), pos),
                ])

        warm = decode()
        print('EAGER_DECODE_PCC', pcc(ttnn.to_torch(warm), expected_decode), flush=True)
        del warm
        restore()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        capturing = True
        if args.inject_capture_failure:
            raise RuntimeError("INJECTED_CAPTURE_FAILURE")
        traced = decode()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        capturing = False
        results = []
        for _ in range(3):
            restore()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            results.append(ttnn.to_torch(traced).clone())
        original_updated_cache = None
        if remapped is not None:
            original_updated_cache = [ttnn.to_torch(t).clone() for t in (state.key, state.value)]
        decode_pcc = pcc(results[-1], expected_decode)
        if control_decode is not None:
            fusion_decode_pcc = pcc(results[-1], control_decode)
            assert fusion_decode_pcc >= .995
            print('FUSED_VS_FUNCTIONAL_DECODE_PCC', fusion_decode_pcc, flush=True)
        deterministic = all(torch.equal(results[0], y) for y in results[1:])
        for host, device in refresh:
            ttnn.copy_host_to_device_tensor(host, device)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        decode2_pcc = pcc(ttnn.to_torch(traced), expected_decode2)
        remap_pcc = None
        remap_cache_pcc = None
        if remapped is not None:
            ttnn.copy(remapped.key, state.key)
            ttnn.copy(remapped.value, state.value)
            for host, device in remap_refresh:
                ttnn.copy_host_to_device_tensor(host, device)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            remap_pcc = pcc(ttnn.to_torch(traced), expected_decode)
            cache_pccs = []
            for tensor, wanted in ((state.key, expected_key), (state.value, expected_value)):
                data = ttnn.to_torch(tensor)
                got = torch.stack([data[new_pt[b, args.seq // 32], :, args.seq % 32:args.seq % 32 + 1]
                                   for b in range(args.batch)])
                cache_pccs.append(pcc(got, wanted))
            remap_cache_pcc = min(cache_pccs)
            for tensor, original in zip((state.key, state.value), original_updated_cache):
                moved_host = ttnn.to_torch(tensor)
                assert torch.equal(moved_host[new_pt.flatten().long()], original[pt_host.flatten().long()]), "Page remapping changed optimized cache contents"
        workload_last_decode_host_us = None
        if args.profile or args.workload:
            restore()
            for step, updates in enumerate(perf_refresh):
                for host, device in updates:
                    ttnn.copy_host_to_device_tensor(host, device)
                if step == 127 and args.profile:
                    ttnn.synchronize_device(mesh)
                    signpost('PERF_DECODE')
                step_started = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                workload_last_decode_host_us = (time.perf_counter() - step_started) * 1e6
                if step == 127 and args.profile:
                    ttnn.synchronize_device(mesh)
                    signpost('PERF_DECODE_END')
                if args.profile and (step + 1) % 16 == 0:
                    ttnn.ReadDeviceProfiler(mesh)  # outside final-step signposts
        workload_final_pcc = None
        if workload_reference is not None:
            workload_final_pcc = pcc(ttnn.to_torch(traced), workload_reference)
            assert workload_final_pcc >= .995, workload_final_pcc
        host_traced_us = None
        if args.timing:
            restore()
            start = time.perf_counter()
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            host_traced_us = (time.perf_counter() - start) * 1e4
            print('SCREEN_HOST_TRACED_US', host_traced_us, flush=True)
        ttnn.release_trace(mesh, tid)
        tid = None
        result = dict(layer_type=ref.layer_type, layer_idx=args.layer, batch=args.batch, sequence=args.seq,
                      weights='synthetic' if args.synthetic else 'real', prefill_pcc=prefill_pcc, decode_pcc=decode_pcc,
                      changed_input_decode_pcc=decode2_pcc, resume_pcc=resume_pcc, deterministic=deterministic)
        result.update(remap_pcc=remap_pcc, remap_cache_pcc=remap_cache_pcc)
        result.update(input_std=args.input_std, request_results=request_results)
        result.update(prefill_pcc_fp32=prefill_pcc_fp32, prefill_relative_rmse=prefill_relative_rmse)
        result['fusion_equivalence'] = dict(prefill_pcc=fusion_prefill_pcc, decode_pcc=fusion_decode_pcc)
        result['screen_host_traced_us'] = host_traced_us
        result['workload_final_decode_pcc'] = workload_final_pcc
        result['workload_last_decode_host_us'] = workload_last_decode_host_us
        result['runtime_fallback_audit'] = not args.profile
        result['provenance'] = provenance
        result['policy'] = getattr(decoder, 'policy', {})
        result['runtime_class'] = type(decoder).__name__
        result['prefill_host_us'] = prefill_host_us
        result['prefill_traced'] = args.trace_prefill
        result['prefill_deterministic'] = prefill_deterministic
        result['activations'] = 'recorded HF preceding layers' if args.recorded else 'synthetic'
        result['memory'] = memory
        result['prefill_compared_tokens'] = expected.shape[1]
        if args.profile or args.workload:
            result['performance_workload'] = dict(input_tokens=4096, output_tokens=128,
                batch=args.batch, concurrency=1, decode_measurement='final warmed step at context 4224',
                scope='single layer, supplied hidden activations')
            result['performance_workload']['prefill_profiled'] = args.profile and not args.decode_profile_only
        print(json.dumps(result), flush=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        assert prefill_pcc >= .995 and decode_pcc >= .995 and decode2_pcc >= .995 and deterministic
        assert resume_pcc is None or resume_pcc >= .995
        assert remap_pcc is None or remap_pcc >= .995
    finally:
        if tid is not None:
            try:
                if capturing:
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.release_trace(mesh, tid)
                print("TRACE_CLEANUP_OK", flush=True)
            except Exception as error:
                print("TRACE_CLEANUP_FAILED", repr(error), flush=True)
        ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
