import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

class G:
    def __init__(s,x,y): s.x=x; s.y=y

grid = G(8,8); total=64
shapes = [(1,1,32,32),(1,1,64,128),(1,1,32,256),(2,4,128,512),(1,1,2048,256),(1,32,1024),(128,512),(1,1,32,4096),(1,1,32,16384),(1,1,64,12288)]
for shp in shapes:
    W=shp[-1]; Wt=W//32
    vol=1
    for d in shp: vol*=d
    Ht=(vol//W)//32
    for hg in [False,True]:
        res=Wt+(Wt if hg else 0)
        row_fits = res<=pd.RESIDENT_BUDGET_TILES
        if Ht>=total and row_fits:
            print(f"ROUTE {shp} gamma={hg}: A (cores={min(Ht,total)})"); continue
        K=pd._select_k(Wt,Ht,grid,total,hg)
        ra=min(Ht,total)
        if K is not None and Ht*K>ra:
            print(f"ROUTE {shp} gamma={hg}: B K={K} cores={Ht*K} Wt_s={Wt//K}")
        else:
            print(f"ROUTE {shp} gamma={hg}: A-fallback (cores={ra})")
