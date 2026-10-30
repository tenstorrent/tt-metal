# SPDX-License-Identifier: Apache-2.0
"""No device, model, network or process launch in these fixtures."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('long_control_host_test',Path(__file__).with_name('agentic_long_control_driver.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def sse(chunks, done=True):
    text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)
    return io.BytesIO((text+('data: [DONE]\n\n' if done else '')).encode())


class LongControlHostTest(unittest.TestCase):
    def test_cached_reference_is_complete_and_pinned(self):
        case,answer,hf,_=driver.reference()
        self.assertEqual(len(case['input_tokens']),32769)
        self.assertEqual(len(answer['tokens']),33)
        self.assertIn(answer['tokens'][-1],hf['eos_ids'])
        self.assertFalse(case['enable_thinking'])

    def test_native_stream_preserves_payload_and_tokens(self):
        chunks=[{'prompt_token_ids':[10,11],'choices':[{'index':0,'delta':{'content':''}}]},
            {'choices':[{'index':0,'delta':{'content':'{"port":7319}'},'token_ids':[90,91],'finish_reason':None}]},
            {'choices':[{'index':0,'delta':{},'token_ids':[248046],'finish_reason':'stop','stop_reason':None}]},
            {'choices':[],'usage':{'prompt_tokens':2,'completion_tokens':3}}]
        messages=[{'role':'user','content':'Keep this native prompt.'}]
        sent=[]
        def fake(request,timeout):
            self.assertEqual(timeout,12000);sent.append(json.loads(request.data));return sse(chunks)
        with tempfile.TemporaryDirectory() as d,patch.object(driver.urllib.request,'urlopen',side_effect=fake):
            row=driver.stream_request('http://unused.invalid',messages,Path(d))
            messages[0]['content']='changed later'
            self.assertEqual(row['status'],'completed')
            self.assertEqual(row['token_ids'],[90,91,248046])
            self.assertEqual(row['prompt_token_ids'],[10,11])
            self.assertEqual(row['content'],'{"port":7319}')
            self.assertEqual(row['request_sha256'],driver.probe.sha(row['request']))
            self.assertEqual(len((Path(d)/'chunks.jsonl').read_text().splitlines()),4)
            self.assertEqual(json.loads((Path(d)/'response.json').read_text())['status'],'completed')
        self.assertEqual(sent[0]['max_tokens'],81920)
        self.assertEqual(sent[0]['chat_template_kwargs'],{'enable_thinking':False})
        self.assertEqual(sent[0]['messages'][0]['content'],'Keep this native prompt.')
        self.assertNotIn('logprobs',sent[0])

    def test_incomplete_stream_keeps_received_tokens(self):
        chunks=[{'choices':[{'index':0,'delta':{'content':'partial'},'token_ids':[7,8]}]}]
        with tempfile.TemporaryDirectory() as d,patch.object(driver.urllib.request,'urlopen',return_value=sse(chunks,False)):
            row=driver.stream_request('http://unused.invalid',[{'role':'user','content':'x'}],Path(d))
            self.assertEqual(row['status'],'error')
            saved=json.loads((Path(d)/'response.json').read_text())
            self.assertEqual(saved['token_ids'],[7,8]);self.assertEqual(saved['content'],'partial')
            self.assertEqual(len((Path(d)/'chunks.jsonl').read_text().splitlines()),1)

    def test_eos_is_never_silently_trimmed(self):
        case={'input_tokens':[1,2], 'expected_facts':{'port':7319}}
        hf={'tokens':[90,91,248046]}
        response={'token_ids':[90,91], 'content':'{"port":7319}', 'usage':{'prompt_tokens':2},
                  'prompt_token_ids':[1,2],'finish_reason':'stop','stop_reason':None,'reasoning':''}
        c=driver.compare_response(response,case,hf,[248046])
        self.assertFalse(c['raw_native_tokens_exact']);self.assertIsNone(c['eos_normalized_exact'])
        self.assertTrue(c['possible_unproven_eos_transport_difference'])
        response['stop_reason']=248046
        c=driver.compare_response(response,case,hf,[248046]);self.assertTrue(c['eos_normalized_exact'])
        response['finish_reason']='length'
        self.assertIsNone(driver.compare_response(response,case,hf,[248046])['eos_normalized_exact'])
        response.update(token_ids=hf['tokens'],finish_reason='stop',stop_reason=None)
        self.assertTrue(driver.compare_response(response,case,hf,[248046])['raw_native_tokens_exact'])

    def test_box_failure_stops_only_owned_lifecycle(self):
        stopped=[];commands=[]
        with tempfile.TemporaryDirectory(dir=driver.REPO) as d:
            root=Path(d)/'arm';state=root/'b32/owned_server.json'
            def helper(action,path):
                self.assertEqual(path,state)
                stopped.append(action)
                if action=='stop':path.unlink()
            owner=SimpleNamespace(CONTAINER='fixture-only',helper=helper,
                in_container=lambda p:'/workspace/'+str(p.relative_to(driver.REPO)))
            def checked(command,deadline,log):
                commands.append(command)
                if '--max-num-seqs' in command:
                    state.write_text('{}')
                if '--probe' in command:
                    raise subprocess.TimeoutExpired(command,1)
            lifecycle_helpers=SimpleNamespace(checked_run=checked,check_port_available=lambda:None)
            class Relay:
                returncode=None
                def __init__(self,command,**kwargs):
                    Path(command[-1]).write_text('{}')
                def poll(self):return self.returncode
                def terminate(self):self.returncode=0
                def wait(self,timeout=None):return self.returncode
                def kill(self):self.returncode=-9
            with patch.object(driver,'module',side_effect=lambda name,file:owner if file=='benchmark_server.py' else lifecycle_helpers), \
                 patch.object(driver.subprocess,'Popen',Relay),patch.object(driver.signal,'signal'):
                rc=driver.lifecycle(SimpleNamespace(output=root,box_hours=.5))
            self.assertEqual(rc,2)
            self.assertEqual(stopped,['inspect','stop'])
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['status'],'incomplete')
            self.assertFalse(summary['owned_server_state_remaining'])
            launch=next(c for c in commands if '--max-num-seqs' in c)
            self.assertEqual(launch[launch.index('--max-num-seqs')+1],'32')
            self.assertEqual(launch[launch.index('--scheduler-block-size')+1],'2048')
            self.assertEqual(launch[launch.index('--shutdown-timeout')+1],'30')


if __name__=='__main__':unittest.main()
