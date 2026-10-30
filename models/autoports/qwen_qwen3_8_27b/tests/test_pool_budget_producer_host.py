# SPDX-License-Identifier: Apache-2.0
"""Execute actual model hook and entire pinned worker sizing/overwrite methods.

External vLLM/TT classes are small host fixtures; no imports of device libraries.
Use QWEN38_PLUGIN_WORKER_SOURCE if the isolated checkout lacks pinned dependency.
"""
import ast
import contextlib
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
PLUGIN=Path(os.environ.get('QWEN38_PLUGIN_WORKER_SOURCE',ROOT.parents[2]/'bringup/dependencies/vllm-tt-plugin/src/vllm_tt_plugin/worker.py'))
def function(path,name,namespace):
    node=next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    for a in node.args.args: a.annotation=None
    node.returns=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]
HOOK=function(ROOT/'tt/generator_vllm.py','get_max_tokens_all_users',{})
class Mamba:
    num_speculative_blocks=0;num_prefill_checkpoint_blocks=0;mamba_cache_mode='align'
class BudgetProducerTests(unittest.TestCase):
    def config(self,override=385,block=2048,batch=32,context=262144):
        return NS(cache_config=NS(num_gpu_blocks_override=override,block_size=block,mamba_block_size=block,mamba_cache_mode='align'),
            model_config=NS(model='Qwen/Qwen3.8-27B',max_model_len=context,original_max_model_len=context,get_sliding_window=lambda:None),
            speculative_config=None,batch=batch)
    def hook(self,c,**kw):
        module=ModuleType('vllm.config');module.get_current_vllm_config_or_none=lambda:c
        with patch.dict(sys.modules,{'vllm.config':module}):
            return HOOK(None,max_model_len=kw.get('context',262144),max_num_seqs=kw.get('batch',32),num_devices=kw.get('devices',4),tt_data_parallel=kw.get('dp',1))
    def test_scoped_budget_matrix(self):
        self.assertEqual(self.hook(self.config()),327680)
        for override in (None,353,384):self.assertEqual(self.hook(self.config(override)),262144)
        self.assertEqual(self.hook(None),262144)
        self.assertEqual(self.hook(self.config(block=4096)),262144)
        for kw in ({'batch':16},{'devices':8},{'dp':2}):self.assertEqual(self.hook(self.config(),**kw),262144)
        self.assertEqual(self.hook(self.config(context=131072),context=131072),131072)
        c=self.config();c.speculative_config=object();self.assertEqual(self.hook(c),262144)
    def run_producer(self,override,block=2048):
        c=self.config(override,block);module=ModuleType('vllm.config');module.get_current_vllm_config_or_none=lambda:c
        model=NS(_HYBRID_KV_CACHE_GROUPS_ENABLED=True,get_max_tokens_all_users=lambda **kw:HOOK(None,**kw))
        namespace=dict(math=math,logger=NS(info=lambda *a:None,warning=lambda *a:None),
            get_tt_data_parallel_size=lambda c:1,get_model_architecture=lambda c:(model,None),get_tt_per_lane_max_num_seqs=lambda c:c.batch,
            get_tt_output_tokens_per_step=lambda c:1,kv_cache_spec_from_model_class=lambda *a:{'fixture':object()},
            _kv_cache_groups_for_sizing=lambda *a:(None,[NS(kv_cache_spec=object())]+[NS(kv_cache_spec=Mamba()) for _ in range(3)]),
            MambaSpec=Mamba,_upstream_admission_blocks=lambda *a:200,is_tt_block_output_model=lambda c:False,
            _fit_block_output_max_model_len=lambda *a:None,
            _available_kv_cache_memory_bytes_for_num_blocks=lambda c,spec,count:count*56311808)
        sizing=function(PLUGIN,'get_num_available_blocks_tt',namespace)
        init=function(PLUGIN,'_init_kv_pool',namespace)
        available=function(PLUGIN,'determine_available_memory',namespace)
        worker=NS(uses_token_kv_cache=True,vllm_config=c,cache_config=c.cache_config,num_devices=4,_num_tt_blocks=None,get_kv_cache_spec=lambda:{})
        with patch.dict(sys.modules,{'vllm.config':module}):
            init(worker);memory=available(worker)
        return worker._num_tt_blocks,c.cache_config.num_gpu_blocks_override,memory
    def test_actual_worker_chain_preserves385_and_baseline(self):
        self.assertEqual(self.run_producer(385),(385,385,385*56311808))
        self.assertEqual(self.run_producer(353),(353,353,353*56311808))
        self.assertEqual(self.run_producer(None),(353,353,353*56311808))
        self.assertEqual(self.run_producer(None,4096)[:2],(289,289))
    def test_old_model_budget_reproduces_actual_cli_loss(self):
        global HOOK
        with patch.dict(globals(),{'HOOK':lambda cls,**kw:kw['max_model_len']}):
            self.assertEqual(self.run_producer(385)[:2],(353,353))
if __name__=='__main__':unittest.main()
