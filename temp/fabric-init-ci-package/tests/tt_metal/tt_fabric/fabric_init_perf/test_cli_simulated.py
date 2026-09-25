"""Local control-flow tests with fake tools, NOT a real Tracy/hardware test."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run as r
from test_runner import fixture, NAMES

class SimulatedCLITests(unittest.TestCase):
    def prepare(self, directory, broken_csv=False, timeout=False):
        root=Path(directory); build=root/'build'
        binary=build/'test/tt_metal/tt_fabric/fabric_init_benchmark'
        capture=build/'tools/profiler/bin/tracy-capture'
        exporter=capture.with_name('tracy-csvexport')
        metadata={
            'schema_version':1,'pid':123,'completed':True,'build_type':'Release','fabric_mode':'FABRIC_2D',
            'phases':{p:{'arch':'blackhole','devices':32,'device_ids':list(range(32)),'shape':[4,8],
                         'teardown_complete':True,'artifacts_before':0 if p=='cold' else 10,'artifacts_after':10,
                         'cache_counters':{'compiled':10 if p=='cold' else 0,'cache_hits':0,'dedup':0}}
                      for p in ('cold','hot')}}
        binary_code=('import time; time.sleep(30)' if timeout else
            "import json,sys,pathlib\nm=json.loads("+repr(json.dumps(metadata))+ ")\nm['audit']='--audit' in sys.argv\n"
            "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(m))\n")
        csv='name,src_file,src_line,ns_since_start,exec_time_ns,thread\n'
        for z in fixture():
            if broken_csv and z.name=='routing':continue
            csv+=f'{z.name},a.cpp,1,{z.start},{z.duration},{z.thread}\n'
        files={binary:binary_code,
               capture:"import sys,pathlib\npathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'FAKE TRACE')\n",
               exporter:'print('+repr(csv)+',end="")\n'}
        for path,body in files.items():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('#!/usr/bin/env python3\n'+body);path.chmod(0o755)
        cfg={'schema_version':1,'measurement_contract':'test','zones':NAMES,
             'hardware':{'hw':{'arch':'blackhole','devices':32,'fabric_mode':'FABRIC_2D'}}}
        (root/'config.json').write_text(json.dumps(cfg))
        (root/'baseline.json').write_text(json.dumps({'schema_version':1,'hardware':'hw','identity':None,'phases':{}}))
        return ['--hardware','hw','--repo',str(root),'--build-dir',str(build),'--config',str(root/'config.json'),
                '--baseline',str(root/'baseline.json'),'--output',str(root/'results'),'--pairs','1','--timeout','1']
    def execute(self,args):
        with patch.dict(os.environ,{'PATH':os.environ['PATH']},clear=True), patch.object(r.subprocess,'check_output',side_effect=['abc\n','']):
            return r.main(args)
    def test_success_reports_and_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.execute(self.prepare(d)),0)
            out=Path(d)/'results'
            self.assertTrue((out/'pair-00/capture.tracy').is_file())
            data=json.loads((out/'comparison.json').read_text());self.assertEqual(len(data['rows']),10)
            self.assertTrue(all(v['status']=='UNBASELINED' for v in data['rows']))
            self.assertIsNone(json.loads((out/'baseline-candidate.json').read_text())['provenance']['approved_by'])
    def test_missing_zone_is_measurement_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.execute(self.prepare(d,broken_csv=True)),2)
            self.assertTrue((Path(d)/'results/error.json').is_file())
    def test_timeout_writes_error_and_cleans_child(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.execute(self.prepare(d,timeout=True)),2)
            self.assertTrue((Path(d)/'results/junit.xml').is_file())
if __name__=='__main__':unittest.main()
