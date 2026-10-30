# SPDX-License-Identifier: Apache-2.0
"""Whole launcher path through owned start, mocked before any process/network/device."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('pool_launch',Path(__file__).with_name('benchmark_server.py'))
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
class AtStart(Exception):pass
class PoolOverrideTest(unittest.TestCase):
    def launch(self,override,block=2048,batch=32,old=None):
        calls=[]; captured=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'observed/server.json';root.joinpath('observed').mkdir()
            state=root/'owned_server.json'
            if old:state.write_text(json.dumps(old))
            def helper(action,state,payload=None):
                calls.append(action)
                if action=='inspect':return {'owner':old['owner']}
                if action=='stop':Path(state).unlink();return {}
                if action=='start':
                    data=json.loads(Path(payload).read_text());server.launch_environment(data)
                    captured.append(data);raise AtStart()
                raise AssertionError(action)
            args=SimpleNamespace(output=str(out),stop=False,max_num_seqs=batch,scheduler_block_size=block,
                num_gpu_blocks_override=override,shutdown_timeout=30,serving_events=None,base_url='http://127.0.0.1:8000')
            with patch.object(server,'helper',side_effect=helper),patch.object(server,'in_container',side_effect=str),patch.object(server.signal,'signal'),patch.object(server.urllib.request,'urlopen',side_effect=OSError('no endpoint')):
                with self.assertRaises(AtStart):server.main(args)
        return captured[0],calls
    def test_full_launcher_records_both_pool_choices(self):
        for value in (353,385):
            with self.subTest(value=value):
                data,calls=self.launch(value)
                self.assertEqual(data['num_gpu_blocks_override'],value)
                self.assertEqual(data['args'][data['args'].index('--num-gpu-blocks-override')+1],str(value))
                self.assertEqual(data['scheduler_block_size'],2048)
                self.assertEqual(calls,['start'])
    def test_default_unchanged(self):
        data,_=self.launch(None,4096)
        self.assertNotIn('num_gpu_blocks_override',data)
        self.assertNotIn('--num-gpu-blocks-override',data['args'])
    def test_changed_pool_cannot_reuse_owned_server(self):
        old={'capacity':32,'scheduler_block_size':2048,'num_gpu_blocks_override':353,
             'shutdown_timeout':30,'owner':{'start_ticks':'1','state':'S'}}
        data,calls=self.launch(385,old=old)
        self.assertEqual(calls,['inspect','stop','start'])
        self.assertEqual(data['num_gpu_blocks_override'],385)
    def test_manifest_argv_mismatch_rejected(self):
        data,_=self.launch(385);data['num_gpu_blocks_override']=353
        with self.assertRaisesRegex(ValueError,'disagrees'):server.launch_environment(data)
    def test_unrecorded_override_rejected(self):
        data,_=self.launch(385);del data['num_gpu_blocks_override']
        with self.assertRaisesRegex(ValueError,'disagrees'):server.launch_environment(data)
    def test_actual_pool_guard_rejects_cli_only_success(self):
        with tempfile.TemporaryDirectory() as d:
            events=Path(d)/'events.jsonl'
            events.write_text(json.dumps({'event':'prepared','allocation':{'scheduler_blocks':353,'scheduler_block_size':2048,'group_count':4}})+'\n')
            with self.assertRaisesRegex(RuntimeError,'vLLM pool'):server.verify_pool_observation(385,'num_gpu_blocks="353"',events)
            with self.assertRaisesRegex(RuntimeError,'Actual four-group'):server.verify_pool_observation(385,'num_gpu_blocks="385"',events)
            events.write_text(json.dumps({'event':'prepared','allocation':{'scheduler_blocks':385,'scheduler_block_size':2048,'group_count':4}})+'\n')
            self.assertEqual(server.verify_pool_observation(385,'num_gpu_blocks="385"',events)['scheduler_blocks'],385)
            with self.assertRaisesRegex(RuntimeError,'vLLM pool'):server.verify_pool_observation(385,'',events)

    def test_wrong_block_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError,'B32/block2048'):self.launch(385,4096)
    def test_wrong_batch_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError,'B32/block2048'):self.launch(385,2048,16)
if __name__=='__main__':unittest.main()
