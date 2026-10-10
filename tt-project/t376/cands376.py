"""t376: per-layer sweep candidates (C_in_block fixed = table, prefetch shard fits), table first, nearest first."""
import sys
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import build_all_blockings, prefetch_shard_fits
from models.tt_dit.utils.conv3d import _BLOCKINGS
L = {"s4res_x": (128,128,147,70,62,(147,68,60)), "s1up_x": (512,4096,39,19,17,(39,17,15)),
     "s0res_x": (1024,1024,21,11,10,(21,9,8)), "s0up_x": (1024,4096,21,11,10,(21,9,8)),
     "s4out_x": (128,48,147,70,62,(147,68,60))}
CAP = 40
for n,(ci,co,T,H,W,key) in L.items():
    tbl = tuple(_BLOCKINGS[(4,8,ci,co,(3,3,3),*key)])
    c = [b for b in build_all_blockings(ci,co,(3,3,3),H,W,T,max_t_block=8,hw_product=(16,32,64))
         if b[0]==tbl[0] and prefetch_shard_fits(*b,(3,3,3),ci)]
    def d(b): return (b[1]!=tbl[1]) + abs(b[2]-tbl[2])/2 + ((b[3],b[4])!=(tbl[3],tbl[4])) + 0.5*(b[3]*b[4]!=tbl[3]*tbl[4])
    c = sorted(set(c), key=lambda b:(d(b),b))[:CAP]
    assert c[0]==tbl
    print(n, ";".join(",".join(map(str,b)) for b in c))
