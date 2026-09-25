import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run as r

NAMES = {'bringup':'bringup','init':'init','routing_tables':'routing','build_compile_configure':'build','configure':'configure'}

def fixture():
    rows = []
    for offset, phase in [(0,'cold'),(1000,'hot')]:
        for name,start,duration in [(f'FabricInitBenchmark::{phase}',0,900),('bringup',10,800),
                                    ('init',20,400),('routing',30,90),('build',130,200),('configure',600,100)]:
            rows.append(r.Zone(name,start+offset,duration,'main'))
    return rows

def baseline(identity, value=100):
    return {'schema_version':1,'hardware':'hw','identity':identity,
            'provenance':{'reference_commit':'abc','approved_by':'reviewer'},
            'phases':{p:{k:{'median_ms':value,'max_regression_percent':10,'min_regression_ms':5}
                         for k in NAMES} for p in r.PHASES}}

def samples(v):
    return [{p:{k:v for k in NAMES} for p in r.PHASES} for _ in range(3)]

class ExtractTests(unittest.TestCase):
    def test_valid_inclusive_durations(self):
        out=r.extract(fixture(),NAMES)
        self.assertEqual(out['cold']['bringup'],0.0008)
        self.assertEqual(out['hot']['init'],0.0004)
    def test_missing_zone_fails(self):
        with self.assertRaises(r.MeasurementError):r.extract(fixture()[:-1],NAMES)
    def test_duplicate_zone_fails(self):
        with self.assertRaises(r.MeasurementError):r.extract(fixture()+[fixture()[1]],NAMES)
    def test_other_thread_fails(self):
        z=fixture();z[3]=r.dataclasses.replace(z[3],thread='worker')
        with self.assertRaises(r.MeasurementError):r.extract(z,NAMES)
    def test_overlapping_callsite_scopes_fail(self):
        z=fixture();z[3]=r.dataclasses.replace(z[3],duration=200)
        with self.assertRaises(r.MeasurementError):r.extract(z,NAMES)
    def test_missing_hot_marker_fails(self):
        with self.assertRaises(r.MeasurementError):r.extract([z for z in fixture() if z.name!='FabricInitBenchmark::hot'],NAMES)
    def test_parent_can_include_dispatch_gap(self):
        r.extract(fixture(),NAMES)
    def test_wrong_phase_boundary_fails(self):
        z=fixture();z[6]=r.dataclasses.replace(z[6],start=100)
        with self.assertRaises(r.MeasurementError):r.extract(z,NAMES)
    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'zones.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['name','src_file','src_line','ns_since_start','exec_time_ns','thread'])
                for z in fixture():w.writerow([z.name,'a.cpp',1,z.start,z.duration,z.thread])
            self.assertEqual(r.extract(r.read_zones(p),NAMES),r.extract(fixture(),NAMES))
    def test_aggregated_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'zones.csv';p.write_text('name,total_ns,counts\na,20,1\n')
            with self.assertRaises(r.MeasurementError):r.read_zones(p)

class ComparisonTests(unittest.TestCase):
    def setUp(self): self.identity={'hardware':'hw'}
    def test_regression(self):
        self.assertTrue(all(x['status']=='REGRESSION' for x in r.compare(samples(111),baseline(self.identity),self.identity,NAMES,'enforce')))
    def test_exact_boundary_passes(self):
        self.assertTrue(all(x['status']=='PASS' for x in r.compare(samples(110),baseline(self.identity),self.identity,NAMES,'enforce')))
    def test_absolute_floor(self):
        self.assertTrue(all(x['status']=='PASS' for x in r.compare(samples(5),baseline(self.identity,1),self.identity,NAMES,'enforce')))
    def test_improvement_passes(self):
        self.assertTrue(all(x['status']=='PASS' for x in r.compare(samples(50),baseline(self.identity),self.identity,NAMES,'enforce')))
    def test_unmeasured_report(self):
        b=baseline(self.identity,None);b['identity']=None
        self.assertTrue(all(x['status']=='UNBASELINED' for x in r.compare(samples(10),b,self.identity,NAMES,'report')))
    def test_unmeasured_enforcement_fails(self):
        with self.assertRaises(r.MeasurementError):r.compare(samples(10),baseline(self.identity,None),self.identity,NAMES,'enforce')
    def test_wrong_identity_fails_even_report(self):
        with self.assertRaises(r.MeasurementError):r.compare(samples(10),baseline({'hardware':'other'}),self.identity,NAMES,'report')
    def test_nan_rejected(self):
        with self.assertRaises(r.MeasurementError):r.compare(samples(float('nan')),baseline(self.identity),self.identity,NAMES,'enforce')
    def test_unapproved_baseline_rejected(self):
        b=baseline(self.identity);b['provenance']['approved_by']=None
        with self.assertRaises(r.MeasurementError):r.compare(samples(10),b,self.identity,NAMES,'enforce')
    def test_zero_baseline_rejected(self):
        with self.assertRaises(r.MeasurementError):r.compare(samples(10),baseline(self.identity,0),self.identity,NAMES,'enforce')
    def test_insufficient_repeats_rejected(self):
        with self.assertRaises(r.MeasurementError):r.compare(samples(10)[:1],baseline(self.identity),self.identity,NAMES,'enforce')
    def test_reports_written(self):
        rows=r.compare(samples(111),baseline(self.identity),self.identity,NAMES,'enforce')
        with tempfile.TemporaryDirectory() as d:
            r.report(Path(d),rows,'enforce',self.identity)
            self.assertEqual(r.ET.parse(Path(d)/'junit.xml').getroot().get('failures'),'10')
    def test_environment_ccache_unset_not_zero(self):
        with patch.dict(r.os.environ,{'TT_METAL_CCACHE_KERNEL_SUPPORT':'0','CCACHE_REMOTE_STORAGE':'remote'},clear=True):
            env=r.controlled_env(Path('/tmp/cache'),Path('/build'),Path('/repo'))
        self.assertNotIn('TT_METAL_CCACHE_KERNEL_SUPPORT',env)
        self.assertNotIn('CCACHE_REMOTE_STORAGE',env)
        self.assertEqual(env['CCACHE_DISABLE'],'1')
    def test_disallow_partial_device_visibility(self):
        with patch.dict(r.os.environ,{'TT_VISIBLE_DEVICES':'0'},clear=True):
            with self.assertRaises(r.MeasurementError):r.controlled_env(Path('/c'),Path('/b'),Path('/r'))

class MetadataTests(unittest.TestCase):
    def metadata(self):
        p={'teardown_complete':True,'arch':'blackhole','devices':32,'shape':[4,8],
           'device_ids':list(range(32)),'artifacts_after':10,'artifacts_before':10,'cache_counters':{'compiled':10,'cache_hits':0,'dedup':0}}
        cold=copy.deepcopy(p);cold['artifacts_before']=0
        hot=copy.deepcopy(p);hot['cache_counters']['compiled']=0
        return {'schema_version':1,'completed':True,'audit':True,'pid':42,'build_type':'Release',
                'fabric_mode':'FABRIC_2D','phases':{'cold':cold,'hot':hot}}
    def hw(self):return {'arch':'blackhole','devices':32,'fabric_mode':'FABRIC_2D'}
    def test_valid_audit(self):self.assertEqual(r.validate_metadata(self.metadata(),self.hw(),True),[4,8])
    def test_cache_not_warm_fails(self):
        m=self.metadata();m['phases']['hot']['cache_counters']['compiled']=10
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)
    def test_cold_did_not_compile_fails(self):
        m=self.metadata();m['phases']['cold']['cache_counters']['compiled']=0
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)
    def test_debug_build_fails(self):
        m=self.metadata();m['build_type']='Debug'
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)
    def test_physical_device_change_fails(self):
        m=self.metadata();m['phases']['hot']['device_ids'][-1]=99
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)
    def test_cold_artifacts_fails(self):
        m=self.metadata();m['phases']['cold']['artifacts_before']=1
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)
    def test_incomplete_teardown_fails(self):
        m=self.metadata();m['phases']['cold']['teardown_complete']=False
        with self.assertRaises(r.MeasurementError):r.validate_metadata(m,self.hw(),True)

if __name__=='__main__':unittest.main()
