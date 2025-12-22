# Elasticsearch Backend Removal - Migration Guide

**Date:** December 22, 2025
**Status:** Elasticsearch backend removed and no longer supported
**Affected Components:** Sweep Test Framework

---

## Summary

Elasticsearch is **no longer supported** as a backend for:
- ✗ Test vector storage
- ✗ Test result storage

**New Architecture:**
- ✓ File-based JSON storage (primary)
- ✓ SFTP upload to data pipeline (for CI/production)
- ✓ PostgreSQL (optional, for advanced features)

---

## Why Was Elasticsearch Removed?

1. **Simplified Architecture**: File-based storage with SFTP upload is more maintainable
2. **Reduced Dependencies**: No need for separate Elasticsearch infrastructure
3. **Security**: Eliminated need for Elasticsearch credentials in environment
4. **Cost**: No Elasticsearch hosting or maintenance overhead
5. **Reliability**: Files are simpler, more portable, and easier to debug

---

## Migration Path

### For Vector Storage

**❌ Old (Elasticsearch)**:
```bash
# Vectors stored in Elasticsearch
export ELASTIC_USERNAME="username"
export ELASTIC_PASSWORD="password"

python tests/sweep_framework/sweeps_runner.py \
  --vector-source elastic \
  --result-dest elastic \
  --module-name my.module
```

**✅ New (File-based)**:
```bash
# Step 1: Generate vectors (stored in JSON files)
python tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name my.module \
  --tag my-tag

# Step 2: Run tests using file-based vectors
python tests/sweep_framework/sweeps_runner.py \
  --vector-source vectors_export \
  --result-dest results_export \
  --module-name my.module
```

**Vector Storage Location:**
- Generated vectors: `tests/sweep_framework/vectors_export/<module_name>.json`
- One JSON file per sweep module
- Human-readable, version-control friendly

### For Result Storage

**❌ Old (Elasticsearch)**:
```bash
export ELASTIC_USERNAME="username"
export ELASTIC_PASSWORD="password"

python tests/sweep_framework/sweeps_runner.py \
  --result-dest elastic
```

**✅ New (File + Optional SFTP Upload)**:
```bash
# Local development - results stored in JSON files
python tests/sweep_framework/sweeps_runner.py \
  --result-dest results_export

# CI/Production - results uploaded to data pipeline via SFTP
python tests/sweep_framework/sweeps_runner.py \
  --result-dest superset
```

**Result Storage Locations:**
- Local results: `tests/sweep_framework/results_export/`
  - Test results: `<module>_<hash>_<timestamp>.json`
  - Run metadata: `oprun_<run_id>.json`
- Superset destination: Automatically uploads `oprun_*.json` via SFTP to data pipeline

---

## Command Line Changes

### Vector Source Options

| Old Options | New Options | Status |
|-------------|-------------|--------|
| `--vector-source elastic` | ❌ **REMOVED** | No longer supported |
| `--vector-source file` | ✅ **KEPT** | Supported |
| `--vector-source vectors_export` | ✅ **KEPT** | Default, recommended |

### Result Destination Options

| Old Options | New Options | Status |
|-------------|-------------|--------|
| `--result-dest elastic` | ❌ **REMOVED** | No longer supported |
| `--result-dest postgres` | ℹ️ **OPTIONAL** | Still available for advanced use |
| `--result-dest results_export` | ✅ **DEFAULT** | Local file storage |
| `--result-dest superset` | ✅ **RECOMMENDED** | CI/production (SFTP upload) |

### Environment Variables

| Variable | Status | Alternative |
|----------|--------|-------------|
| `ELASTIC_USERNAME` | ❌ **REMOVED** | Not needed |
| `ELASTIC_PASSWORD` | ❌ **REMOVED** | Not needed |
| `POSTGRES_*` | ✅ **OPTIONAL** | Only if using PostgreSQL |

---

## Data Migration

### Historical Elasticsearch Data

**Q: What happens to data already in Elasticsearch?**
A: Historical data remains in Elasticsearch (read-only). Contact the data team if you need to query or export historical results.

**Q: Can I migrate old vectors from Elasticsearch to files?**
A: Yes, regenerate vectors using the parameter generator:

```bash
# Regenerate vectors for a module
python tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name <module_name> \
  --tag <your_tag>

# This creates: tests/sweep_framework/vectors_export/<module_name>.json
```

### New Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sweep Test Workflow                       │
└─────────────────────────────────────────────────────────────┘

1. GENERATE VECTORS
   sweeps_parameter_generator.py
   ↓
   vectors_export/<module>.json  (Local JSON files)

2. RUN TESTS
   sweeps_runner.py --vector-source vectors_export
   ↓
   results_export/<module>_*.json  (Local JSON files)
   results_export/oprun_*.json     (Run metadata)

3. UPLOAD (Optional - CI/Production only)
   --result-dest superset
   ↓
   SFTP Upload → Data Pipeline → Airflow → PostgreSQL → Superset
```

---

## Updated Workflows

### Local Development Workflow

```bash
# 1. Generate test vectors
python tests/sweep_framework/sweeps_parameter_generator.py \
  --module-name eltwise.unary.relu.relu

# 2. Run tests locally
python tests/sweep_framework/sweeps_runner.py \
  --module-name eltwise.unary.relu.relu \
  --vector-source vectors_export \
  --result-dest results_export

# 3. View results
ls -lh tests/sweep_framework/results_export/
cat tests/sweep_framework/results_export/oprun_*.json | jq
```

### CI Workflow

```bash
# CI workflow automatically uses:
--vector-source vectors_export   # Vectors from previous generation step
--result-dest superset            # Auto-upload to data pipeline
```

See `.github/workflows/ttnn-run-sweeps.yaml` for details.

---

## Troubleshooting

### Issue: "Unknown vector source: elastic"

**Error:**
```
ValueError: Unknown vector source: elastic. Supported sources: 'file', 'vectors_export'
```

**Solution:**
Change `--vector-source elastic` to `--vector-source vectors_export`

### Issue: "Unknown result destination: elastic"

**Error:**
```
ValueError: Unknown result destination: elastic. Supported destinations: 'results_export', 'superset'
```

**Solution:**
Change `--result-dest elastic` to `--result-dest results_export` (or `superset` for CI)

### Issue: "ELASTIC_USERNAME and ELASTIC_PASSWORD must be set"

**Error:**
```
ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in environment variables
```

**Solution:**
This error should not occur after the migration. If you see it:
1. Update your tt-metal repository: `git pull`
2. Verify you're using the latest code (post-December 2025)
3. Remove elastic-related environment variables from your shell

### Issue: "Cannot find vectors for module"

**Solution:**
Vectors must be generated first:
```bash
python tests/sweep_framework/sweeps_parameter_generator.py --module-name <module_name>
```

### Issue: "Where are my test results?"

**Solution:**
Results are in `tests/sweep_framework/results_export/`:
```bash
# List all results
ls -lh tests/sweep_framework/results_export/

# View run metadata
cat tests/sweep_framework/results_export/oprun_*.json | jq

# View specific test results
cat tests/sweep_framework/results_export/<module>_*.json | jq
```

---

## FAQ

### Q: Why was Elasticsearch removed?

**A:** The Elasticsearch backend added complexity, required credentials management, and had hosting costs. File-based storage with SFTP upload to the data pipeline is simpler, more secure, and easier to maintain.

### Q: What about historical data in Elasticsearch?

**A:** Historical data remains available (read-only). Contact the data team if you need to query or export historical results. New data flows through the file → SFTP → data pipeline → Superset path.

### Q: Can I still use Elasticsearch?

**A:** No, Elasticsearch support has been completely removed from the codebase. All references, configuration, credentials, and backend code have been deleted.

### Q: Do I need to regenerate all my test vectors?

**A:** Yes, if you were using Elasticsearch for vector storage. Run the parameter generator for each module you need:

```bash
python tests/sweep_framework/sweeps_parameter_generator.py --module-name <module_name>
```

This creates JSON files in `vectors_export/` directory.

### Q: Where do results go in CI?

**A:** CI uses `--result-dest superset`, which:
1. Saves results locally to `results_export/`
2. Automatically uploads `oprun_*.json` files via SFTP to the data pipeline
3. Data flows: SFTP → Airflow → PostgreSQL → Superset dashboards

### Q: Can I still query results programmatically?

**A:** Yes, but the method has changed:

**Old:** Query Elasticsearch
**New:** Parse JSON files from `results_export/` or query PostgreSQL (if results were uploaded via superset destination)

### Q: What if I encounter issues?

**A:**
1. Check this migration guide
2. Review `tests/sweep_framework/README.md` for updated examples
3. Run validation: `pytest tests/sweep_framework/test_no_elasticsearch.py`
4. Contact the TT-Metal team for support

---

## Validation

To verify Elasticsearch has been completely removed:

```bash
# Run validation tests
cd tests/sweep_framework
pytest test_no_elasticsearch.py -v

# Expected output: 9 passed
```

All tests should pass, confirming:
- ✓ No Elasticsearch imports
- ✓ elastic_config.py deleted
- ✓ ElasticVectorSource removed
- ✓ ElasticResultDestination removed
- ✓ Elasticsearch removed from requirements
- ✓ Factory classes reject "elastic" option

---

## Timeline

- **December 22, 2025**: Elasticsearch backend removed (Phase 0 complete)
- **Going forward**: File-based storage (vectors_export, results_export) is the only supported method

---

## Additional Resources

- **Sweep Framework Documentation**: `tests/sweep_framework/README.md`
- **CI Workflow**: `.github/workflows/ttnn-run-sweeps.yaml`
- **Code Changes**: See Phase 0 tasks in `tests/sweep_framework/plan.md`
- **Validation Tests**: `tests/sweep_framework/test_no_elasticsearch.py`

---

## Summary Checklist

After migrating, you should:

- [ ] Remove `ELASTIC_USERNAME` and `ELASTIC_PASSWORD` from environment
- [ ] Update scripts to use `--vector-source vectors_export`
- [ ] Update scripts to use `--result-dest results_export` (or `superset` for CI)
- [ ] Regenerate test vectors using parameter generator
- [ ] Update any custom tooling or queries to use JSON files
- [ ] Remove any Elasticsearch queries or integrations from your workflows

---

**Questions or Issues?**
Contact the TT-Metal team or file an issue in the repository.
