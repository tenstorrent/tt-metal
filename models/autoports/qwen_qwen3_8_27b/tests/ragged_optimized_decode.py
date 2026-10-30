# SPDX-License-Identifier: Apache-2.0
"""Real-weight traced full attention with different current positions per request."""
import json
import os
from pathlib import Path

import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from .run_optimized_decoder import pcc, reference_prefill, audit_runtime
from ..tt.optimized_decoder import OptimizedDecoder


def main():
    torch.set_num_threads(8)
    torch.manual_seed(719)
    evidence = Path(__file__).resolve().parents[1] / 'doc/functional_decoder'
    cfg = AutoConfig.from_pretrained(str(evidence / 'hf_config.json')).text_config
    cfg._attn_implementation = 'eager'
    weights = load_file('.cache/qwen38-weights/layer_3.safetensors')
    ref = Qwen3_5DecoderLayer(cfg, 3).to(torch.bfloat16).eval()
    ref.load_state_dict({k.removeprefix('model.language_model.layers.3.'): v for k, v in weights.items()})
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    recorded = torch.load(".cache/qwen38-weights/recorded_layer_3.pt", weights_only=True)
    positions = torch.tensor([1, 31, 32, 127, 128, 129, 511, 513])[torch.randperm(8)]
    fixtures, expected, expected2 = [], [], []
    with torch.no_grad():
        for position in positions.tolist():
            x = recorded[:, :position]
            xd = recorded[:, position:position+1]
            xd2 = recorded[:, position+1:position+2]
            cs = rope(x, torch.arange(position).expand(3, 1, -1))
            ds = rope(xd, torch.full((3, 1, 1), position, dtype=torch.long))
            ds2 = rope(xd2, torch.full((3, 1, 1), position + 1, dtype=torch.long))
            cache = DynamicCache(config=cfg)
            reference_prefill(ref, x, *cs, cache)
            expected.append(ref(xd, position_embeddings=ds, past_key_values=cache))
            expected2.append(ref(xd2, position_embeddings=ds2, past_key_values=cache))
            fixtures.append((x, xd, xd2, cs, ds, ds2))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    trace = None
    capturing = False
    try:
        def put(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh):
            return ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, device=device)
        decoder = OptimizedDecoder.from_state_dict(weights, hf_config=cfg, layer_idx=3, mesh_device=mesh, policy=json.loads(os.environ.get("QWEN_OPT_POLICY", "{}")))
        decoder.prefill_forward = audit_runtime(decoder.prefill_forward)
        decoder.decode_forward = audit_runtime(decoder.decode_forward)
        state = decoder.allocate_state(batch_size=8, context_length=768)
        page_table = torch.randperm(192, dtype=torch.int32).reshape(8, 24)
        for b, (x, _, _, cs, _, _) in enumerate(fixtures):
            decoder.prefill_forward(put(x), state=state,
                page_table=put(page_table[b:b+1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                cos=put(cs[0][:, None]), sin=put(cs[1][:, None]))
        backup = decoder.snapshot(state)
        pt = put(page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = put(positions.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        x = put(torch.cat([f[1] for f in fixtures]))
        cos = put(torch.cat([f[4][0][:, None] for f in fixtures]))
        sin = put(torch.cat([f[4][1][:, None] for f in fixtures]))
        updates = [(put(torch.cat([f[2] for f in fixtures]), device=None), x),
                   (put(torch.cat([f[5][0][:, None] for f in fixtures]), device=None), cos),
                   (put(torch.cat([f[5][1][:, None] for f in fixtures]), device=None), sin),
                   (put((positions+1).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=None), pos)]
        def restore():
            ttnn.copy(backup.key, state.key)
            ttnn.copy(backup.value, state.value)
        def decode():
            return decoder.decode_forward(x, state=state, page_table=pt, current_pos=pos, cos=cos, sin=sin)
        decode()
        restore()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capturing = True
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capturing = False
        outputs = []
        for _ in range(3):
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            outputs.append(ttnn.to_torch(out))
        score = pcc(outputs[-1], torch.cat(expected))
        for host, device in updates:
            ttnn.copy_host_to_device_tensor(host, device)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        score2 = pcc(ttnn.to_torch(out), torch.cat(expected2))
        ttnn.release_trace(mesh, trace)
        trace = None
        result = dict(layer_type='full_attention', batch=8, positions=positions.tolist(),
                      decode_pcc=score, changed_position_pcc=score2,
                      deterministic=all(torch.equal(outputs[0], y) for y in outputs[1:]), runtime_fallback_audit=True)
        (evidence.parent / 'optimized_decoder/ragged_decode.json').write_text(json.dumps(result, indent=2)+'\n')
        print(result, flush=True)
        assert score >= .995 and score2 >= .995 and result['deterministic']
    finally:
        if trace is not None:
            if capturing:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
