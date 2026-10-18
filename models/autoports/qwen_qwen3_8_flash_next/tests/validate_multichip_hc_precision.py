"""Isolate precision lost at tensor-parallel hyperconnection boundaries."""
import argparse,sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--hc-policy',choices=['read','norm','all'],default='all');a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    factory=target.MultichipDecoder.from_state_dict.__func__;linear=ttnn.linear;reduce=target.MeshOps.reduce;norm=target.MeshOps.norm;weights=set()
    def create(cls,*args,**kw):
        obj=factory(cls,*args,**kw)
        weights.update(h.read_weight.buffer_address() for h in (obj.attn_hyper,obj.mlp_hyper))
        return obj
    def mm(x,w,**kw):
        if a.hc_policy in ('read','all') and w.buffer_address() in weights:kw['dtype']=ttnn.float32
        return linear(x,w,**kw)
    def sum_partials(self,x):
        out=reduce(self,x)
        return ttnn.typecast(out,ttnn.bfloat16) if x.dtype==ttnn.float32 and x.shape[-1]==352 else out
    def exact_norm(self,x,w,eps):
        if a.hc_policy not in ('norm','all'):return norm(self,x,w,eps)
        stats = ttnn.rms_norm_pre_all_gather(x,dtype=ttnn.float32,compute_kernel_config=self.compute)
        y = ttnn.rms_norm_post_all_gather(x,self.gather(stats),epsilon=eps,compute_kernel_config=self.compute,dtype=ttnn.float32)
        return ttnn.typecast(ttnn.multiply(y,w),x.dtype)
    target.MultichipDecoder.from_state_dict=classmethod(create);ttnn.linear=mm;target.MeshOps.reduce=sum_partials;target.MeshOps.norm=exact_norm
    from models.autoports.qwen_qwen3_8_flash_next.tests.validate_multichip_decoder import main
    main()
if __name__=='__main__':main()
