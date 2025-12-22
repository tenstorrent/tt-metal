# Phase 0, Task 0.9: Security Cleanup - Administrator Action Required

## GitHub Secrets Cleanup

The following GitHub Actions secrets are **no longer needed** and should be removed by a repository administrator:

### Secrets to Delete:
- `SWEEPS_ELASTIC_USERNAME`
- `SWEEPS_ELASTIC_PASSWORD`

### How to Remove (Requires Admin Access):

1. Navigate to: `https://github.com/tenstorrent/tt-metal/settings/secrets/actions`
2. Locate the secrets listed above
3. Click "Remove" for each secret
4. Confirm deletion

### Why Remove These Secrets?

1. **No Longer Used**: Elasticsearch backend has been completely removed
2. **Security Best Practice**: Remove unused credentials
3. **Reduced Attack Surface**: Fewer secrets = fewer potential security issues
4. **Clear Signal**: Removal confirms Elasticsearch is fully deprecated

### Verification

After removal, verify the secrets are gone:
- Check GitHub Actions secrets page
- Confirm CI workflows run successfully without these secrets

### Local Environment Cleanup (Already Completed)

✅ No `ELASTIC_USERNAME` or `ELASTIC_PASSWORD` in shell rc files
✅ No credentials in `.env` or `.envrc` files
✅ Updated `tests/README.md` to remove credential examples

---

**Status**: Local cleanup complete ✅
**Action Required**: Administrator must delete GitHub secrets
**Date**: December 22, 2025
