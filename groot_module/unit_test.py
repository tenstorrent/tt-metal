import ttnn
import torch

def test_layout(device):
    t1 = torch.randn((32,1572864),dtype=torch.bfloat16)
    ttnn_t1 = ttnn.from_torch(t1,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
    ttnn_t1_rm = ttnn.to_layout(ttnn_t1,layout=ttnn.ROW_MAJOR_LAYOUT)