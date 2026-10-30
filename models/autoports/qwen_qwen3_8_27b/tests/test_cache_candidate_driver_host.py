# SPDX-License-Identifier: Apache-2.0
"""Mocked candidate lifecycle; never opens ports, runs Docker or starts children."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

PATH=Path(__file__).with_name('agentic_batch_driver.py')
spec=importlib.util.spec_from_file_location('cache_driver_test',PATH)
DRIVER=importlib.util.module_from_spec(spec)
spec.loader.exec_module(DRIVER)

class CandidateDriverTests(unittest.TestCase):
    def exercise(self, cutoff, pool=False):
        with tempfile.TemporaryDirectory(dir=DRIVER.REPO,prefix='.candidate-driver-test-') as tmp:
            root=Path(tmp)/'fresh'
            baseline={'path':'/retained/paired-02/b32/probe/result.json','sha256':'receipt-only','model_revision':'pinned-checkpoint'}
            commands=[];cleanup=[]
            def helper(action,state):
                cleanup.append(action)
                if action=='stop':state.unlink()
            fake_server=NS(helper=helper)
            def run(command,deadline,log_path,cleanup_reserve=90):
                commands.append(command)
                self.assertGreater(deadline,DRIVER.time.monotonic())
                if command[1].endswith('benchmark_server.py'):
                    (root/'b32/owned_server.json').write_text('{}')
                    (root/'b32/observed/server.json').write_text(json.dumps({
                        'model_revision':'pinned-checkpoint','tokenizer_revision':'pinned-checkpoint',
                        'layer_count':64,'max_num_seqs':32,
                        'server_command':['--block-size','2048','--mamba-block-size','2048'] + (['--num-gpu-blocks-override','385'] if pool else [])}))
                else:
                    output=root/'b32/probe';output.mkdir()
                    (output/'partial-response.json').write_text('{"status":"completed"}')
                    if cutoff:raise TimeoutError('simulated effort box')
                    (output/'result.json').write_text(json.dumps({'status':'review_required',
                        'invariant_evidence_complete':True,'all_native_token_ids_present':True,
                        'pair':{'exact':True},'measured_wall_seconds':100}))
            class Relay:
                returncode=None
                def poll(self):return self.returncode
                def terminate(self):self.returncode=0
                def wait(self,timeout=None):return self.returncode
                def kill(self):self.returncode=-9
            def popen(command,**kwargs):
                Path(command[-1]).write_text('{}')
                return Relay()
            fake_spec=NS(loader=NS(exec_module=lambda module:None))
            with patch.object(DRIVER,'candidate_reference',return_value=baseline), \
                 patch.object(DRIVER,'check_port_available'), \
                 patch.object(DRIVER,'checked_run',side_effect=run), \
                 patch.object(DRIVER.subprocess,'Popen',side_effect=popen), \
                 patch.object(DRIVER.signal,'signal'), \
                 patch.object(DRIVER.importlib.util,'spec_from_file_location',return_value=fake_spec), \
                 patch.object(DRIVER.importlib.util,'module_from_spec',return_value=fake_server):
                status=DRIVER.main(NS(output=root,candidate_block_size=2048,
                    reference=Path(baseline['path']),candidate_pool_blocks=385 if pool else None,box_hours=.3 if pool else 1))
            self.assertEqual(status,2 if cutoff else 0)
            self.assertEqual(len(commands),2)
            launch,probe=commands
            for option,expected in [('--max-num-seqs','32'),('--scheduler-block-size','2048'),('--shutdown-timeout','30')]:
                self.assertEqual(launch[launch.index(option)+1],expected)
            if pool:
                self.assertEqual(launch[launch.index('--num-gpu-blocks-override')+1],'385')
            for option,expected in [('--batch','32'),('--output-cap','81920'),('--request-timeout','12000'),('--reference',baseline['path'])]:
                self.assertEqual(probe[probe.index(option)+1],expected)
            self.assertEqual(cleanup,['inspect','stop'])
            self.assertTrue((root/'b32/probe/partial-response.json').exists())
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['batches'],[32]);self.assertEqual(summary['owned_server_states_remaining'],[])
            self.assertEqual(summary['relay_exit'],0)
            self.assertFalse(summary['scored'])
            if cutoff:self.assertIn('partial probe is unscored',summary['cutoff_policy'])

    def test_pool_candidate_success(self):self.exercise(False,True)
    def test_pool_candidate_cutoff(self):self.exercise(True,True)

    def test_candidate_only_success_keeps_default_limits(self):self.exercise(False)
    def test_timeout_preserves_partial_responses_and_owned_cleanup(self):self.exercise(True)

if __name__=='__main__':unittest.main()
