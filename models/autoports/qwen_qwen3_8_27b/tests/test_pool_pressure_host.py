# SPDX-License-Identifier: Apache-2.0
"""Host-only pressure manifest/exactness guards; no hardware, processes or network."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('pool_pressure_test',Path(__file__).with_name('agentic_pool_capacity.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
class PressureTests(unittest.TestCase):
    def cohort(self):
        rows=[dict(agent=i,input_tokens=[99,i]+[7]*59998) for i in range(10)]
        for r in rows:r['input_sha256']=driver.probe.sha(r['input_tokens'])
        return dict(revision=driver.long.REVISION,cases=rows,cases_sha256=driver.probe.sha(rows))
    def test_distinct_prefix_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.json';cohort=self.cohort();p.write_text(json.dumps(cohort))
            self.assertEqual(len(driver.cases(p)['cases']),10)
            cohort['cases'][1]['input_tokens'][1]=0
            cohort['cases'][1]['input_sha256']=driver.probe.sha(cohort['cases'][1]['input_tokens'])
            cohort['cases_sha256']=driver.probe.sha(cohort['cases']);p.write_text(json.dumps(cohort))
            with self.assertRaisesRegex(ValueError,'Shared complete'):driver.cases(p)
    def test_corrupted_native_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.json';cohort=self.cohort();cohort['cases'][0]['input_tokens'][5]=8
            cohort['cases_sha256']=driver.probe.sha(cohort['cases']);p.write_text(json.dumps(cohort))
            with self.assertRaisesRegex(ValueError,'native input'):driver.cases(p)
    def pair(self):
        row=dict(request_sha256='request',prompt_token_ids=[1,2],token_ids=[3,248046],status='completed',finish_reason='stop')
        return dict(cases_sha256='cohort',rounds=[dict(responses=[copy.deepcopy(row) for _ in range(10)]) for _ in range(2)])
    def test_exact_pair_detects_each_material_difference(self):
        a=self.pair();self.assertTrue(driver.compare_pair(a,a))
        for key,value in [('request_sha256','different'),('prompt_token_ids',[1,4]),('token_ids',[3]),('status','error'),('finish_reason','length')]:
            b=copy.deepcopy(a);b['rounds'][1]['responses'][4][key]=value
            self.assertFalse(driver.compare_pair(a,b),key)
    def test_incomplete_pair_rejected(self):
        a=self.pair();b=copy.deepcopy(a);b['rounds'][1]['responses'].pop()
        self.assertFalse(driver.compare_pair(a,b))
    def probe_fixture(self, final_error=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cohort=self.cohort()
            for r in cohort['cases']:
                r.update(messages=[{'role':'user','content':'agent '+str(r['agent'])}],expected={'agent':r['agent']})
            cohort['cases_sha256']=driver.probe.sha(cohort['cases']);cp=root/'cases.json';cp.write_text(json.dumps(cohort))
            identity=root/'identity.json';identity.write_text(json.dumps(dict(layer_count=64,max_num_seqs=32,model_revision=driver.long.REVISION,server_command=['--block-size','2048','--mamba-block-size','2048','--num-gpu-blocks-override','353'])))
            events=root/'events.jsonl';events.write_text(json.dumps({'event':'prepared','allocation':{'proof':'fixture'}})+'\n')
            seen=[]
            def stream(url,messages,out):
                if out.name=='warmup':return {'status':'completed','finish_reason':'stop'}
                i=int(out.name.removeprefix('agent'));seen.append((str(out.parent),i,len(messages)))
                return dict(status='error' if final_error and len(messages)==3 and i==9 else 'completed',stream_done=not(final_error and len(messages)==3 and i==9),usage={'completion_tokens':2},finish_reason='stop',token_ids=[100+i,248046],prompt_token_ids=cohort['cases'][i]['input_tokens'],content=json.dumps({'agent':i}),request_sha256=driver.probe.sha(messages))
            args=NS(cases=cp,server_config=identity,pool_blocks=353,probe=root/'probe',base_url='http://unused',events=events,reference=None)
            with patch.object(driver.long,'stream_request',side_effect=stream),patch.object(driver.probe,'event_snapshot',return_value={}),patch.object(driver.probe,'metric_snapshot',return_value=['vllm:cache_config_info{num_gpu_blocks="353"} 1']),patch.object(driver.probe,'counter_delta',return_value=0):
                self.assertEqual(driver.run_probe(args),2 if final_error else 0)
            result=json.loads((root/'probe/result.json').read_text())
            self.assertEqual(result['confirmation_complete'],not final_error);self.assertEqual(len(seen),20)
            self.assertEqual(sorted(n for _,_,n in seen),[1]*10+[3]*10)
            self.assertEqual(result['max_tokens'],81920);self.assertEqual(result['request_timeout_seconds'],12000)

    def test_full_two_wave_probe_preserves_native_semantic_eos_and_pool(self):self.probe_fixture()
    def test_final_error_after_json_eos_never_passes(self):self.probe_fixture(final_error=True)

    def test_candidate_cannot_launch_without_pressure_reference(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.json';p.write_text(json.dumps(self.cohort()))
            with self.assertRaisesRegex(ValueError,'completed353 reference'):
                driver.lifecycle(NS(cases=p,pool_blocks=385,reference=None))
if __name__=='__main__':unittest.main()
