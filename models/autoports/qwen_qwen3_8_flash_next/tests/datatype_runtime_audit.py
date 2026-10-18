"""Observe native arguments during preparation/capture without retaining tensors."""
from collections import Counter
import json
import ttnn

class RuntimePolicyAudit:
    def __init__(self, model):
        self.phase='warmup'
        self.rows=Counter()
        self.layers={}
        self.originals=[]
        self.weights={}
        def bind(obj, names, prefix):
            for name in names:
                value=getattr(obj,name,None)
                if value is not None:self.weights[id(value)]=f'{prefix}.{name}'
        bind(model,('lm_head',),'terminal')
        bind(model.terminal,('down','up'),'terminal_mixer')
        for index,l in zip(model.layer_indices,model.layers):
            prefix=f'layer{index}'
            bind(l.attention,('packed_weight','packed_projection','decode_projection'),prefix+'.attention')
            bind(l.attention.output_projection,('row_weight','column_weight'),prefix+'.output')
            bind(l.attn_hyper,('read_weight','up'),prefix+'.attn_hc')
            bind(l.mlp_hyper,('read_weight','up'),prefix+'.mlp_hc')
            bind(l.mlp,('router','shared_gu','shared_down'),prefix+'.mlp')
            self.weights[id(l.mlp.gates[0])]=prefix+'.routed_experts'
            if l.ple is not None:bind(l.ple,('projection',),prefix+'.ple')

    def record(self, op, args, kwargs):
        row={'phase':self.phase,'op':op}
        if op=='linear':
            x,w=args[:2]
            row.update(group=self.weights.get(id(w),'other'),input_shape=list(x.shape),weight_shape=list(w.shape),input_dtype=str(x.dtype),weight_dtype=str(w.dtype),output_dtype=str(kwargs.get('dtype','inherit')))
        elif op=='unified_routed_expert_moe':
            row.update(group=self.weights.get(id(args[4][0]),'other'),input_dtype=str(args[0].dtype),capacity=args[7],
                       gate_dtype=str(args[4][0].dtype),up_dtype=str(args[5][0].dtype),down_dtype=str(args[6][0].dtype),local_experts=len(args[4]))
        elif op=='sparse_sdpa':
            row.update(q_dtype=str(args[0].dtype),kv_dtype=str(args[1].dtype),kv_format=str(kwargs.get('kv_format')),q_shape=list(args[0].shape),kv_shape=list(args[1].shape))
        else:
            row.update(input_dtype=str(args[0].dtype),input_shape=list(args[0].shape))
        config=kwargs.get('compute_kernel_config')
        if config is not None:
            row['compute']={k:str(getattr(config,k)) for k in ('math_fidelity','math_approx_mode','fp32_dest_acc_en','packer_l1_acc')}
        label=row.get("group", "")
        layer=None
        if label.startswith("layer"):
            prefix,row["group"]=label.split(".",1)
            layer=int(prefix[5:])
        key=json.dumps(row,sort_keys=True)
        self.rows[key]+=1
        if layer is not None:self.layers.setdefault(key,set()).add(layer)

    def patch(self, namespace, name, replacement):
        self.originals.append((namespace,name,getattr(namespace,name)))
        setattr(namespace,name,replacement)

    def __enter__(self):
        for namespace,name in ((ttnn,'linear'),(ttnn.experimental,'all_gather_async'),
                               (ttnn.experimental,'reduce_scatter_minimal_async'),
                               (ttnn.experimental.deepseek_prefill,'unified_routed_expert_moe'),
                               (ttnn.transformer,'sparse_sdpa')):
            original=getattr(namespace,name)
            def wrapper(*args,_original=original,_name=name,**kwargs):
                self.record(_name,args,kwargs)
                return _original(*args,**kwargs)
            self.patch(namespace,name,wrapper)
        begin,end=ttnn.begin_trace_capture,ttnn.end_trace_capture
        def begin_capture(*args,**kwargs):
            result=begin(*args,**kwargs)
            self.phase='capture'
            return result
        def end_capture(*args,**kwargs):
            result=end(*args,**kwargs)
            self.phase='warmup'
            return result
        self.patch(ttnn,'begin_trace_capture',begin_capture)
        self.patch(ttnn,'end_trace_capture',end_capture)
        return self

    def __exit__(self,*exc):
        for namespace,name,original in reversed(self.originals):setattr(namespace,name,original)

    def report(self):
        return [dict(json.loads(key),calls=count,layers=sorted(self.layers.get(key,()))) for key,count in self.rows.items()]
