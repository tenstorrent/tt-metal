import importlib.util, json, pathlib, subprocess, sys, tempfile, unittest

P=pathlib.Path(__file__).with_name("sfpu_corpus.py")
S=importlib.util.spec_from_file_location("sfpu_corpus",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

class CorpusTest(unittest.TestCase):
    def test_inventory_and_no_substring_mapping(self):
        rows={r["id"]:r for r in M.inventory()}
        self.assertEqual(len(rows),164)
        self.assertEqual(rows["legacy__ckernel_sfpu_clamp"]["mapping_state"],"unmapped")
        self.assertEqual(rows["legacy__ckernel_sfpu_comp"]["mapping_state"],"unmapped")

    def test_comparator(self):
        base={"id":"x","arch":"bh","metric":"device_cycles","scope":"body","selector":"generated","cycles":100}
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/"b.json"; p.write_text(json.dumps({"results":[base]}))
            def result(c=None):
                r=dict(base); r["cycles"]=c; return M.compare_baseline([r],p,2)[0]["status"]
            self.assertEqual(result(99),"PASS")
            self.assertEqual(result(103),"REGRESSION")
            self.assertEqual(result(None),"SKIP_NO_DEVICE_CYCLES")
            current=dict(base); current["cycles"]=103
            c=pathlib.Path(d)/"c.json"; c.write_text(json.dumps({"results":[current]}))
            self.assertNotEqual(subprocess.run([sys.executable,str(P),"--compare-results",str(c),"--baseline",str(p),"--max-regression-pct","2"],stdout=subprocess.DEVNULL).returncode,0)

if __name__=="__main__": unittest.main()
