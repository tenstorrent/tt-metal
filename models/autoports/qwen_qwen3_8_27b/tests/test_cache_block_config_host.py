# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only host checks; execute actual pure adapter methods without TT imports."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CONFIG = load(ROOT / 'tt/cache_block_config.py')
LAUNCH = load(ROOT / 'tests/benchmark_server.py')

def actual_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]

ADAPTER = ROOT / 'tt/generator_vllm.py'

class Table:
    def __init__(self, rows): self.rows = rows
    def __getitem__(self, key): return self.rows[key[0]][key[1]]

class CacheBlockTests(unittest.TestCase):
    def test_default_and_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(CONFIG.scheduler_block_size(), 4096)
        for value in ('2048', '4096'):
            with patch.dict(os.environ, {CONFIG.BLOCK_SIZE_ENV:value}):
                self.assertEqual(CONFIG.scheduler_block_size(), int(value))
        for value in ('1024', '2049', ' 2048', '', '2048.0'):
            with patch.dict(os.environ, {CONFIG.BLOCK_SIZE_ENV:value}):
                with self.assertRaises(ValueError): CONFIG.scheduler_block_size()

    def test_launch_env_matches_both_cli_flags_and_overrides_ambient(self):
        for size in (2048,4096):
            payload = dict(phase_path='/tmp/phases',scheduler_block_size=size,
                args=['--block-size',str(size),'--mamba-block-size',str(size)])
            with patch.dict(os.environ, {CONFIG.BLOCK_SIZE_ENV:'invalid'}):
                self.assertEqual(LAUNCH.launch_environment(payload)[CONFIG.BLOCK_SIZE_ENV], str(size))
            payload['args'][-1] = '1024'
            with self.assertRaises(ValueError): LAUNCH.launch_environment(payload)

    def test_specs_and_mismatch_rejection(self):
        class Attention(NS):
            @property
            def page_size_bytes(self): return self.block_size*self.num_kv_heads*self.head_size*2*2
        class Mamba(NS): pass
        fn = actual_function(ROOT/'tt/vllm_cache_specs.py','get_kv_cache_spec',dict(
            scheduler_block_size=CONFIG.scheduler_block_size,FullAttentionSpec=Attention,
            MambaSpec=Mamba,torch=NS(bfloat16='bf16',float32='fp32'),
            MambaAttentionBackendEnum=NS(GDN_ATTN='gdn'),CONV_SHAPE=(3,2560),RECURRENT_SHAPE=(12,128,128)))
        for size in (2048,4096):
            with patch.dict(os.environ,{CONFIG.BLOCK_SIZE_ENV:str(size)}):
                config=NS(cache_config=NS(block_size=size,mamba_block_size=size,mamba_cache_mode='align'),
                    model_config=NS(hf_config=NS(text_config=NS(num_key_value_heads=8,head_dim=256,layer_types=['linear_attention','full_attention']))))
                specs=fn(config); state=specs['model.layers.0.state']; attention=specs['model.layers.1.state']
                self.assertEqual(state.block_size, size)
                self.assertEqual(state.prefill_checkpoint_alignment,size)
                self.assertEqual(state.page_size_padded,attention.page_size_bytes)
                self.assertGreaterEqual(state.page_size_padded,801792)
                config.cache_config.mamba_block_size=1024
                with self.assertRaises(ValueError):fn(config)
                config.cache_config.mamba_block_size=size
                config.cache_config.block_size=1024
                with self.assertRaises(ValueError):fn(config)

    def test_snapshot_columns_null_rejection_and_decode_crossings(self):
        snapshot=actual_function(ADAPTER,'_snapshot_ids',{})
        save=actual_function(ADAPTER,'_save_decode_boundary',{})
        for size in (2048,4096):
            obj=NS(scheduler_block_size=size,state_pool=NS(pools={0:None,3:None}),
                generator=NS(_decode_active=NS(tolist=lambda:[True,False])),slot_ends=[size-1,128])
            tables={0:Table([[11,12,13],[21,22,23]]),3:Table([[31,32,33],[41,42,43]])}
            obj._snapshot_ids=lambda *args:snapshot(obj,*args)
            saved=[];obj.state_pool.save=lambda *args:saved.append(args)
            for n in (1,2,3):self.assertEqual(snapshot(obj,tables,0,n*size),{0:10+n,3:30+n})
            for end in (0,size-1,size+1):
                with self.assertRaises(ValueError):snapshot(obj,tables,0,end)
            save(obj,[size-1,99999],True,tables)
            self.assertEqual(saved,[(0,{0:11,3:31})]);self.assertEqual(obj.slot_ends,[size,128])
            save(obj,None,False,tables)
            self.assertEqual(len(saved),1);self.assertEqual(obj.slot_ends,[size+1,128])
            tables[3].rows[0][0]=0
            with self.assertRaises(ValueError):snapshot(obj,tables,0,size)

    def test_actual_page_expansion_has_no_cross_block_alias(self):
        class Matrix:
            def __init__(self, rows): self.rows=rows
            @property
            def shape(self): return (len(self.rows),len(self.rows[0]))
            def to(self, dtype): return self
            def unsqueeze(self, dim): return self
            def __mul__(self, count): return Matrix([[v*count for v in row] for row in self.rows])
            def __add__(self, offsets): return NS(flatten=lambda dim:Matrix([[v+x for v in row for x in offsets] for row in self.rows]))
            def __setitem__(self, key, value):
                for r,row in enumerate(value.rows): self.rows[r][:len(row)]=row
        torch=NS(int32=None,arange=lambda count,dtype:range(count),
            zeros_like=lambda src:Matrix([[0]*src.shape[1] for _ in src.rows]))
        expand=actual_function(ADAPTER,'_page_table',{'torch':torch})
        for size in (2048,4096):
            count=size//32
            obj=NS(scheduler_block_size=size,generator=NS(page_table_host=Matrix([[0]*8192,[0]*8192])))
            table=expand(obj,Matrix([[0,7,9]]))
            self.assertEqual(table.rows[0][:count],list(range(count)))
            self.assertEqual(table.rows[0][count:count*2],list(range(7*count,8*count)))
            self.assertEqual(table.rows[0][count*2:count*3],list(range(9*count,10*count)))
            self.assertEqual(table.rows[1],[0]*8192)
            # Max context remains representable and overflowing scheduler tables reject.
            exact=Matrix([list(range(262144//size))])
            self.assertEqual(expand(obj,exact).rows[0],list(range(8192)))
            with self.assertRaises(ValueError):expand(obj,Matrix([list(range(262144//size+1))]))

    def test_allocation_rejects_env_spec_drift_before_allocating(self):
        import builtins
        def importing(name, *args, **kwargs):
            if name=='serving_state': return NS(ServingStatePool=None)
            if name=='cache_block_config': return CONFIG
            return builtins.__import__(name,*args,**kwargs)
        allocate=actual_function(ADAPTER,'allocate_hybrid_kv_cache',{
            '__builtins__':dict(vars(builtins),__import__=importing),
            'torch':NS(bfloat16='bf16')})
        class ReachedAllocation(Exception): pass
        for size in (2048,4096):
            with patch.dict(os.environ,{CONFIG.BLOCK_SIZE_ENV:str(size)}):
                shapes=[]
                def allocating(shape,*args):
                    shapes.append(shape)
                    raise ReachedAllocation()
                obj=NS(model=NS(layers=[None]),allocate_kv_cache=allocating)
                group=NS(kv_cache_spec=NS(block_size=size),layer_names=['model.layers.0.state'])
                cfg=NS(kv_cache_groups=[group],num_blocks=353 if size==2048 else 289)
                with self.assertRaises(ReachedAllocation): allocate(obj,cfg)
                self.assertEqual(shapes,[(cfg.num_blocks*(size//32),1,32,256)])
                self.assertEqual(obj.scheduler_block_size,size)
                group.kv_cache_spec.block_size=1024
                with self.assertRaises(ValueError):allocate(obj,cfg)
                self.assertEqual(len(shapes),1)

    def test_entire_hybrid_allocation_and_evidence_for_both_sizes(self):
        import builtins
        class Pool:
            def __init__(self, model, states, num_blocks_by_layer):
                self.pools = dict(num_blocks_by_layer)
                self.allocated_bytes_per_device = sum(num_blocks_by_layer.values())*801792
        def importing(name, *args, **kwargs):
            if name=='serving_state': return NS(ServingStatePool=Pool)
            if name=='cache_block_config': return CONFIG
            return builtins.__import__(name,*args,**kwargs)
        memory=NS(total_bytes_allocated_per_bank=10000,total_bytes_free_per_bank=2000,num_banks=8)
        allocate=actual_function(ADAPTER,'allocate_hybrid_kv_cache',{
            '__builtins__':dict(vars(builtins),__import__=importing),
            'torch':NS(bfloat16='bf16'),
            'ttnn':NS(BufferType=NS(DRAM='dram'),get_memory_view=lambda mesh,kind:memory)})
        for size,blocks in ((2048,353),(4096,289)):
            with self.subTest(block_size=size), patch.dict(os.environ,{CONFIG.BLOCK_SIZE_ENV:str(size)}):
                states=[NS(conv=object(),recurrent=object()),NS(key=NS(shape=(blocks*(size//32),1,32,256)))]
                calls=[]
                def allocating(shape,dtype,num_layers):
                    calls.append((shape,dtype,num_layers));return states
                model=NS(layer_indices=[0,3],layers=[NS(layer_type='linear_attention'),NS(layer_type='full_attention')],precision_config={'test':'unchanged'})
                obj=NS(model=model,mesh=object(),batch_size=32,allocate_kv_cache=allocating)
                cfg=NS(num_blocks=blocks,kv_cache_groups=[
                    NS(kv_cache_spec=NS(block_size=size),layer_names=['model.layers.0.state']),
                    NS(kv_cache_spec=NS(block_size=size),layer_names=['model.layers.3.state'])])
                self.assertIs(allocate(obj,cfg),states)
                self.assertEqual(calls,[((blocks*(size//32),1,32,256),'bf16',2)])
                self.assertEqual(obj.cache_groups,{0:0,3:1})
                self.assertEqual(obj.state_pool.pools,{0:blocks})
                evidence=obj.allocation_evidence
                self.assertEqual(evidence['scheduler_block_size'],size)
                self.assertEqual(evidence['scheduler_blocks'],blocks)
                self.assertEqual(evidence['recurrent_snapshot_bytes_per_device'],blocks*801792)
                self.assertEqual(evidence['attention_bytes_per_device'],2*blocks*(size//32)*8*1088)
                self.assertEqual(evidence['active_recurrent_bytes_per_device'],32*801792)
                self.assertEqual(evidence['allocated_bytes_per_device'],80000)
                self.assertEqual(evidence['free_bytes_per_device'],16000)
                self.assertEqual(evidence['selected_layers'],[0,3])
                self.assertEqual(evidence['precision_config'],{'test':'unchanged'})

if __name__=='__main__': unittest.main()
