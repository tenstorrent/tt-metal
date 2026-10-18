"""Separate same-input attention projections, measured with their complete split/concat costs."""
import ttnn


def install(*,dram=False,readers=1,core_count=5,block=4):
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as mod
    from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import configure,linear as dram_linear
    for cls,packed_name,widths in [(mod.OptimizedGatedDeltaNet,'packed_weight',(10240,6144,64,64)),
                                   (mod.OptimizedQSA,'packed_projection',(6144,512,512,6144,640))]:
        old=cls.from_state_dict.__func__; original_linear=cls._linear
        def factory(cls,*args,_old=old,_name=packed_name,_widths=widths,**kwargs):
            obj=_old(cls,*args,**kwargs); w=getattr(obj,_name)
            obj.split_weights=[]; start=0
            for i,width in enumerate(_widths):
                value=w[:,start:start+width]
                setattr(obj,f'split_{i}',value);obj.split_weights.append(value);start+=width
            obj.packed_identity=w.buffer_address()
            if dram:
                names=[f'split_{i}' for i in range(len(_widths))]
                configure(obj,names,cores=[core_count]*len(names),blocks=[block]*len(names),readers=readers)
            return obj
        def linear(self,x,w,_orig=original_linear):
            if w.buffer_address()!=self.packed_identity:
                return _orig(self,x,w)
            values=[dram_linear(self,x,v) if dram else _orig(self,x,v) for v in self.split_weights]
            return ttnn.concat(values,dim=-1)
        cls.from_state_dict=classmethod(factory);cls._linear=linear
