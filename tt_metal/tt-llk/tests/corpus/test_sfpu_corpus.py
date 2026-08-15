import importlib.util, json, pathlib, subprocess, sys, tempfile, unittest

P=pathlib.Path(__file__).with_name("sfpu_corpus.py")
S=importlib.util.spec_from_file_location("sfpu_corpus",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)

class CorpusTest(unittest.TestCase):
    def test_inventory_and_no_substring_mapping(self):
        rows={r["id"]:r for r in M.inventory()}
        self.assertEqual(len(rows),164)
        self.assertEqual(rows["legacy__ckernel_sfpu_clamp"]["mapping_state"],"unmapped")
        self.assertEqual(rows["legacy__ckernel_sfpu_comp"]["mapping_state"],"unmapped")
        self.assertEqual(rows["legacy__ckernel_sfpu_clamp"]["semantic_cpp_class"],"unmapped")
        self.assertEqual(rows["legacy__ckernel_sfpu_comp"]["semantic_cpp_class"],"unmapped")

    def test_every_row_has_a_complete_explicit_semantic_audit(self):
        rows=M.read_manifest(); errors,_=M.validate(rows)
        self.assertEqual(errors,[])
        self.assertEqual(len(rows),164)
        for row in rows:
            self.assertEqual(row["version"],"2")
            self.assertIn(row["semantic_cpp_class"],M.SEMANTIC_CPP_CLASSES)
            self.assertTrue(row["semantic_cpp_blocker"])
            self.assertIn(row["correctness_metric"],M.CORRECTNESS_METRICS)
            if row["mapping_state"] == "mapped":
                self.assertNotEqual(row["semantic_cpp_class"],"unmapped")

    def test_measured_silicon_is_correctness_gated(self):
        rows={r["id"]:r for r in M.read_manifest()}
        expected={
            "legacy__ckernel_sfpu_welfords":"win",
            "legacy__ckernel_sfpu_reduce_custom":"win",
            "legacy__ckernel_sfpu_binary_bcast":"parity",
            "legacy__ckernel_sfpu_where":"loss",
            "legacy__ckernel_sfpu_mul_int":"loss",
            "metal__ckernel_sfpu_mul_int32":"loss",
            "legacy__ckernel_sfpu_topk":"blocked",
        }
        for row_id,status in expected.items():
            row=rows[row_id]; self.assertEqual(row["silicon_status"],status)
            if status in {"win","parity","loss"}:
                self.assertEqual(row["test_status"],"pass")
                self.assertNotEqual(row["correctness_metric"],"none")

    def test_plan_formats_are_machine_and_human_readable(self):
        row=M.read_manifest()[0]
        import contextlib, io
        for fmt in ("tsv","json","markdown"):
            out=io.StringIO()
            with contextlib.redirect_stdout(out): M.emit_plan([row],"bh",fmt)
            self.assertIn(row["id"],out.getvalue())
        payload=json.loads(self._capture_plan([row],"bh","json"))
        self.assertEqual(payload["schema"],2)
        self.assertIn("correctness_threshold",payload["rows"][0])

    @staticmethod
    def _capture_plan(rows,arch,fmt):
        import contextlib, io
        out=io.StringIO()
        with contextlib.redirect_stdout(out): M.emit_plan(rows,arch,fmt)
        return out.getvalue()

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

    def test_checked_in_tsv_baseline(self):
        rows=M.load_baseline(M.DEVICE_BASELINE)
        ids={r["id"] for r in M.inventory()}
        self.assertTrue(rows)
        self.assertTrue({r["id"] for r in rows} <= ids)
        current={"id":"legacy__ckernel_sfpu_welfords","arch":"bh","metric":"device_cycles",
                 "scope":"WELFORD_BODY","selector":"generated","cycles":324}
        result=M.compare_baseline([current],M.DEVICE_BASELINE,0)[0]
        self.assertEqual(result["status"],"REGRESSION")
        self.assertAlmostEqual(result["delta_pct"],100.0/323.0)

    def test_v1_is_retained_as_an_immutable_migration_source(self):
        old=P.with_name("sfpu_corpus_v1.tsv")
        with old.open() as f:
            rows=list(__import__("csv").DictReader((x for x in f if not x.startswith("#")),delimiter="\t"))
        self.assertEqual(len(rows),164)
        self.assertNotIn("semantic_cpp_class",rows[0])
        self.assertTrue(M.MANIFEST.name.endswith("_v2.tsv"))

    def test_empty_mapped_compile_lane_cannot_report_green(self):
        with tempfile.TemporaryDirectory() as d:
            run=pathlib.Path(d)/"qsr"
            result=subprocess.run([sys.executable,str(P),"--mode","compile","--arch","qsr","--execute",
                                   "--require-executed-mapped","--run-root",str(run)],stdout=subprocess.DEVNULL)
            self.assertNotEqual(result.returncode,0)
            data=json.loads((run/"results.json").read_text())
            self.assertEqual(data["provenance"]["executed_mapped_gate"],"FAIL")

if __name__=="__main__": unittest.main()
