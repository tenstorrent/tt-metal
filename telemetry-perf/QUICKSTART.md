# Quick Start Guide - Telemetry Benchmark Suite

## 🚀 What to Run

All files are now in: `telemetry-perf/`

### Option 1: Run the Full Reduced Suite (RECOMMENDED FIRST RUN)

This validates the core hypothesis and provides actionable results in ~2-3 hours:

```bash
cd telemetry-perf
./run_benchmark.sh reduced
```

**What it does:**
- Tests 2, 4, and 8 device configurations
- Validates that `--mmio-only` prevents ERISC contention
- Tests 6 polling frequencies
- Generates comprehensive report
- **Duration: ~2-3 hours**

### Option 2: Run Just the Hypothesis Test (FASTEST)

If you just want to validate the core hypothesis (~30 minutes):

```bash
cd telemetry-perf
./run_benchmark.sh hypothesis
```

**What it does:**
- Tests ONLY: baseline vs --mmio-only vs full mode
- AllGather and ReduceScatter on 2, 4, 8 devices
- Answers: "Does --mmio-only prevent ERISC contention?"
- **Duration: ~30 minutes**

### Option 3: Full Comprehensive Suite

For publication-quality analysis (~9-12 hours):

```bash
cd telemetry-perf
./run_benchmark.sh full
```

---

## 📊 What You'll Get

After running, you'll see results in `/tmp/`:

### Main Report
**`/tmp/telemetry_final_report_reduced.md`** ← **READ THIS FIRST**

Contains:
- Executive summary
- Core hypothesis validation result
- Performance impact analysis
- Production recommendations

### Example Expected Outcome:

```
✓ HYPOTHESIS VALIDATED
  - MMIO-only mode shows <5% impact on all tests
  - Full mode shows ≥10% impact OR failures on all tests

RECOMMENDATION: Use --mmio-only flag for multi-chip telemetry

Recommended command:
./build/tools/tt-telemetry/tt-telemetry \
  --mmio-only \
  --logging-interval 100ms \
  --port 7070
```

### Detailed Data Files
- `/tmp/mmio_validation_results.json` - Core hypothesis test raw data
- `/tmp/single_device_results_reduced.json` - Single-device impact data
- `/tmp/multi_device_results_reduced.json` - Multi-device impact data
- `/tmp/sustained_workload_results.json` - Long-running drift analysis

---

## 🔍 Interpreting Results

### If Hypothesis is VALIDATED ✓

```
✓ HYPOTHESIS VALIDATED
  - MMIO-only shows <5% impact
  - Full mode shows >10% impact or failures
```

**Meaning:** The `--mmio-only` flag successfully prevents ERISC contention.

**Action:** Use `--mmio-only` for all multi-chip deployments.

### If Hypothesis is NOT VALIDATED ✗

```
✗ HYPOTHESIS NOT VALIDATED
  - Some tests did not meet expected criteria
```

**Meaning:** Results don't clearly show MMIO-only advantage.

**Action:** Review individual test results in JSON files to understand why.

---

## 📝 Additional Commands

### Analyze Existing Results

If benchmarks already completed:

```bash
./run_benchmark.sh analyze --phase reduced
```

Generates additional analysis report in `/tmp/telemetry_analysis_summary_reduced.md`

### Run Individual Test Suites

```bash
# Single-device only
./run_benchmark.sh single --phase reduced

# Multi-device only
./run_benchmark.sh multi --phase reduced

# Sustained workload only
./run_benchmark.sh sustained
```

---

## 🛠️ Requirements

### Hardware
- **T3000 system** (8x Wormhole devices) for full testing
- Or **minimum 2x Wormhole devices** for reduced testing

### Software
- tt-metal built with telemetry: `build/tools/tt-telemetry/tt-telemetry`
- Python 3.8+
- Required packages: numpy, scipy, statsmodels

### Build Telemetry

If not already built:

```bash
cd /data/kkfernandez/tt-metal
./build.sh
```

Verify telemetry binary exists:

```bash
ls -lh build/tools/tt-telemetry/tt-telemetry
```

---

## ⚡ Quick Decision Tree

**Want to know if --mmio-only works?**
→ `./run_benchmark.sh hypothesis` (~30 min)

**Want comprehensive validation with recommendations?**
→ `./run_benchmark.sh reduced` (~2-3 hours)

**Need publication-quality data?**
→ `./run_benchmark.sh full` (~9-12 hours)

**Already ran tests, need more analysis?**
→ `./run_benchmark.sh analyze --phase reduced`

---

## 📖 More Information

See **`TELEMETRY_BENCHMARK_README.md`** for:
- Detailed methodology
- Statistical methods explanation
- Troubleshooting guide
- Interpreting advanced metrics
- Customizing test configurations

---

## 🚨 Troubleshooting

### "Devices not available"

```bash
# Reset all devices
for i in {0..3}; do tt-smi -r $i; done
sleep 30
```

### "Port 7070 already in use"

```bash
# Kill existing telemetry
pkill tt-telemetry
sleep 5
```

### "Out of memory"

This shouldn't happen with the default config (max 578MB ~5% DRAM), but if it does:
- Edit test scripts to reduce tensor sizes
- Or use only reduced phase

---

## 💡 Tips

1. **Start with hypothesis test** to quickly validate approach (~30 min)
2. **Then run reduced suite** if hypothesis validates (~2-3 hours)
3. **Save results** - they're in `/tmp/` which may be cleared on reboot
4. **Run during off-hours** - tests are long-running
5. **Check intermediate results** - Scripts save `*_partial.json` every 10 tests

---

## Expected Timeline

| Test | Duration | What It Validates |
|------|----------|-------------------|
| Hypothesis | ~30 min | Core: --mmio-only prevents ERISC contention |
| Reduced Suite | ~2-3 hours | Comprehensive impact across frequencies |
| Full Suite | ~9-12 hours | Publication-quality statistical analysis |
| Individual Tests | ~20-60 min each | Specific workload categories |

---

## Success Criteria

After running tests, look for:

✅ **Core hypothesis validated**
✅ **MMIO-only shows <5% impact**
✅ **Full mode shows >10% impact**
✅ **No significant drift in sustained workload**
✅ **Clear production recommendation provided**

If all ✅ → Use `--mmio-only` with confidence!

---

**Ready to start?**

```bash
cd telemetry-perf
./run_benchmark.sh reduced
```

Go get coffee ☕ - see you in 2-3 hours with results!
