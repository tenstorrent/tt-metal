import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
def pcc(a,b):
    a=a.flatten().float(); b=b.flatten().float()
    return torch.corrcoef(torch.stack([a,b]))[0,1].item()
dev = ttnn.open_device(device_id=0)
try:
    for shape in [(1,1,4096,64),(1,4,4096,64),(1,8,4096,128),(2,4,256,64)]:
        B,H,S,D=shape
        torch.manual_seed(1)
        Q=torch.randn(B,H,S,D,dtype=torch.bfloat16); K=torch.randn(B,H,S,D,dtype=torch.bfloat16); V=torch.randn(B,H,S,D,dtype=torch.bfloat16)
        exp=torch.nn.functional.scaled_dot_product_attention(Q.float(),K.float(),V.float())
        td=lambda t: ttnn.from_torch(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out=ttnn.to_torch(scaled_dot_product_attention(td(Q),td(K),td(V))).float()
        print(f"shape {shape} PCC {pcc(exp,out):.5f}")
finally:
    ttnn.close_device(dev)
