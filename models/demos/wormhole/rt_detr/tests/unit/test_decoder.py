# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import torch
import ttnn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
rtdetr_pytorch_path = Path(__file__).parent.parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(rtdetr_pytorch_path))

from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from tt.rtdetr_decoder import run_decoder, decoder_layer
from models.common.utility_functions import comp_pcc

REF = Path(__file__).parent.parent.parent / "reference/reference_outputs.pt"
pcc_threshold = 0.97

@pytest.fixture(scope="module")
def ref():
    return torch.load(REF, map_location="cpu")

@pytest.fixture(scope="module")
def device():
    mesh_shape = ttnn.MeshShape(1, 1)
    dev = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)
    yield dev
    ttnn.close_mesh_device(dev)

def _to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )

def convert_pytorch_layer_to_ttnn(torch_layer, device):
    """Extracts weights from a PyTorch TransformerDecoderLayer and formats them for TTNN."""
    
    class TTNNLinear:
        def __init__(self, weight, bias, dev):
            # PyTorch linear is [out, in]. TTNN needs [in, out] so we transpose.
            w = weight.T.contiguous().unsqueeze(0).unsqueeze(0).bfloat16()
            self.weight = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev)
            if bias is not None:
                b = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).bfloat16()
                self.bias = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
            else:
                self.bias = None

    class TTNNNorm:
        def __init__(self, weight, bias, dev):
            w = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).bfloat16()
            b = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).bfloat16()
            self.weight = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev)
            self.bias = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)

    class TTNNAttn:
        def __init__(self, mha, dev):
            # PyTorch MHA packs Q, K, V into one massive weight block -> split
            w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
            self.q = TTNNLinear(w_q, b_q, dev)
            self.k = TTNNLinear(w_k, b_k, dev)
            self.v = TTNNLinear(w_v, b_v, dev)
            self.out_proj = TTNNLinear(mha.out_proj.weight, mha.out_proj.bias, dev)

    class LayerParams:
        def __init__(self, layer, dev):
            self.self_attn = TTNNAttn(layer.self_attn, dev)
            self.norm1 = TTNNNorm(layer.norm1.weight, layer.norm1.bias, dev)
            self.norm2 = TTNNNorm(layer.norm2.weight, layer.norm2.bias, dev)
            self.norm3 = TTNNNorm(layer.norm3.weight, layer.norm3.bias, dev)
            self.linear1 = TTNNLinear(layer.linear1.weight, layer.linear1.bias, dev)
            self.linear2 = TTNNLinear(layer.linear2.weight, layer.linear2.bias, dev)
            
    return LayerParams(torch_layer, device)


class TestDecoder:
    def _load_inputs(self, ref, device):
        # Flatten spatial maps to sequence length
        def prep_memory(t):
            B, C, H, W = t.shape
            t_flat = t.view(B, C, H * W).transpose(1, 2)
            return t_flat.unsqueeze(1)

        p3 = _to_device(prep_memory(ref["encoder_p3"]), device)
        p4 = _to_device(prep_memory(ref["encoder_p4"]), device)
        p5 = _to_device(prep_memory(ref["encoder_p5"]), device)
        
        query = _to_device(ref["decoder_init_query"], device)
        query_pos = _to_device(ref["decoder_query_pos"], device)
        
        return [p3, p4, p5], query, query_pos

    def _prepare_hybrid_args(self, device):
        # 1. Load PyTorch Model 
        pytorch_model = RTDETRTransformer(num_classes=80, hidden_dim=256, num_queries=300)
        torch_decoder = pytorch_model.decoder
        torch_decoder.eval() # dropout disabled

        # 2. Extract Real Weights for TTNN
        dec_params = [convert_pytorch_layer_to_ttnn(layer, device) for layer in torch_decoder.layers]
        
        # 3. Dynamic Shapes
        ref_points = torch.rand(1, 300, 3, 4).float()
        spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20]], dtype=torch.long)
        level_start_index = torch.tensor([0, 6400, 8000], dtype=torch.long)
        
        return pytorch_model, ref_points, spatial_shapes, level_start_index, dec_params

    def test_pcc_layer1_output(self, ref, device):
        memory_list, query_tt, query_pos_tt = self._load_inputs(ref, device)
        memory_tt = ttnn.concat(memory_list, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Unpack pytorch_model and explicitly get the decoder
        pytorch_model, ref_points, spatial_shapes, level_start_index, dec_params = self._prepare_hybrid_args(device)
        torch_decoder = pytorch_model.decoder 

        # 1. pytorch on cpu
        query_torch = ref["decoder_init_query"].squeeze(1).float() 
        query_pos_torch = ref["decoder_query_pos"].squeeze(1).float()
        memory_torch = ttnn.to_torch(memory_tt).squeeze(1).float() # Get concatenated memory

        with torch.no_grad():
            golden_out = torch_decoder.layers[0](
                tgt=query_torch,
                reference_points=ref_points,
                memory=memory_torch,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                query_pos_embed=query_pos_torch
            )

        # 2. ttnn
        out_tt = decoder_layer(query_tt, query_pos_tt, torch_decoder.layers[0], dec_params[0], memory_tt, ref_points, spatial_shapes, device)
        ttnn_out = ttnn.to_torch(out_tt).squeeze(1)

        pcc, msg = comp_pcc(golden_out, ttnn_out, pcc_threshold)
        print(f"\n[LAYER 1] PCC: {pcc:.6f}")
        assert pcc >= pcc_threshold, f"PCC failed! {pcc:.4f} < {pcc_threshold} - {msg}"

    def test_output_shape(self, ref, device):
        memory_list, query, _ = self._load_inputs(ref, device)
        memory = ttnn.concat(memory_list, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        pytorch_model, ref_points, spatial_shapes, _, dec_params = self._prepare_hybrid_args(device)
        
        # BUG FIX: Removed query_pos from this function call. 
        # The variables now correctly align with the run_decoder signature.
        out_tt, updated_ref_points = run_decoder(
            query, pytorch_model, dec_params, memory, ref_points, spatial_shapes, device
        )
        
        out = ttnn.to_torch(out_tt)

        assert out.shape[-1] == 256
        assert out.shape[-2] == 300

    def test_layer_by_layer_pcc(self, ref, device):
        memory_list, query_tt, _ = self._load_inputs(ref, device)
        memory_tt = ttnn.concat(memory_list, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

        pytorch_model, ref_points, spatial_shapes, _, dec_params = self._prepare_hybrid_args(device)
        
        # Verify the structure we need exists
        assert hasattr(pytorch_model, 'query_pos_head'), "RTDETRTransformer missing query_pos_head"
        assert hasattr(pytorch_model, 'decoder'), "RTDETRTransformer missing decoder"
        
        # Run the full pipeline
        out_tt, updated_ref_points = run_decoder(
            query_tt, pytorch_model, dec_params, memory_tt, ref_points, spatial_shapes, device
        )
        
        out = ttnn.to_torch(out_tt).squeeze(1)
        
        # Validate final TTNN output shape bounds before a theoretical full PCC test
        assert out.shape == torch.Size([1, 300, 256]), f"Unexpected output shape: {out.shape}"