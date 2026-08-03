"""Real-weight differential oracle for the shipped advisor candidate."""
import json
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX, _config, _page_table, _positions, _real_state, _to_torch_decode, _to_tt_decode,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder


def main():
    batch = 32
    cfg = _config()
    state = _real_state()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        common = dict(hf_config=cfg, layer_idx=LAYER_IDX, mesh_device=device, batch=batch, max_context=64)
        incumbent = OptimizedDecoder.from_state_dict(state, **common)
        candidate = OptimizedDecoder.from_state_dict(
            state, **common, optimization_policy=OptimizationPolicy(advisor_rope_l1="query_key", advisor_norm_cores=11)
        )
        hidden = (torch.randn(batch, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(9432)) * 0.2).bfloat16()
        tt_hidden = _to_tt_decode(hidden, device)
        positions = _positions(list(range(1, batch + 1)), device)
        page_table = _page_table(batch, 64, device, permute=True)

        def run(decoder):
            key_cache, value_cache = decoder.create_paged_kv_cache()
            return _to_torch_decode(decoder.decode_forward(
                tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table,
                current_positions=positions, use_long_rope=False,
            ))

        expected, actual = run(incumbent), run(candidate)
        passed, message = comp_pcc(expected.float(), actual.float(), 0.999999)
        record = {
            "candidate": "rope_l1_query_key_norm_11c", "decode_batch": batch,
            "oracle_weights": "real", "oracle_reference": "frozen incumbent with identical real weights and inputs",
            "oracle_pcc_bar": 0.999999, "oracle_passed": bool(passed), "oracle_message": message,
        }
        out = os.environ["CHALLENGER_ORACLE_OUT"]
        with open(out, "w") as fh:
            json.dump(record, fh, indent=2)
        print(json.dumps(record, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
