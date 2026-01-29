# Docker Build Scripts

This directory contains installation scripts for external tools used in the TT-Metalium Docker images.

## Architecture

The Dockerfile uses **multi-stage builds** to optimize layer caching and reduce rebuild times:

### Layer Types

1. **Official Image Layers** (`uv-layer`)
   - Uses `COPY --from=` with SHA256-pinned official images
   - Zero build overhead - just copies pre-built binaries
   - Most reliable and fastest method
   - See: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv

2. **Simple Binary Layers** (`ccache-layer`, `mold-layer`)
   - Built from `alpine:3.19` (minimal image)
   - Downloads pre-built binaries, verifies SHA256 hash, extracts
   - Zero dependencies on the base Ubuntu image
   - Can be cached independently - changing base image doesn't invalidate these layers

3. **Builder Stages** (`doxygen-builder`, `cba-builder`, `gdb-builder`)
   - Built from `base` Ubuntu image (needs build tools)
   - Downloads source/binary, verifies hash, builds, installs to `/staging`
   - Only the final build artifacts are copied to target images
   - Changes to base image may invalidate these, but build results are still cached

### Benefits

- **Faster rebuilds**: Tool layers don't rebuild when unrelated files change
- **Smaller images**: Only final binaries copied, not build dependencies
- **Parallel builds**: Docker can build independent stages concurrently
- **Hash validation**: All downloads verified before use
- **Reproducible builds**: SHA256 pinning ensures exact same binaries

## Scripts

| Script | Purpose |
|--------|---------|
| `install-ccache.sh` | Install ccache binary release |
| `install-mold.sh` | Install mold linker binary release |
| `install-doxygen.sh` | Build and install doxygen |
| `install-clangbuildanalyzer.sh` | Build and install ClangBuildAnalyzer |
| `install-gdb.sh` | Build and install GDB from source |
| `compute-hashes.sh` | Utility to compute SHA256 hashes for updates |

## Updating Tool Versions

### Updating uv

1. Find the new version's SHA256 digest at: https://github.com/astral-sh/uv/pkgs/container/uv/versions
2. Look for the distroless image (tagged as `X.Y.Z`, `latest`)
3. Update both `UV_VERSION` and the `FROM ghcr.io/astral-sh/uv@sha256:...` line in Dockerfile

Example:
```dockerfile
# Update the version comment
ARG UV_VERSION=0.9.27

# Update the FROM line with new digest
FROM ghcr.io/astral-sh/uv@sha256:NEW_DIGEST_HERE AS uv-layer
```

### Updating other tools

1. Update the version ARG in the Dockerfile
2. Run `compute-hashes.sh` to get the new SHA256 hash
3. Update the corresponding SHA256 ARG in the Dockerfile
4. Test the build

```bash
# Compute hashes for current versions
./dockerfile/scripts/compute-hashes.sh

# Compute hash for a specific version
CCACHE_VERSION=4.11.0 ./dockerfile/scripts/compute-hashes.sh
```

## Hash Verification Sources

- **uv**: GitHub Container Registry page (SHA256 digest shown for each version)
- **ccache**: GPG signatures available at release page
- **mold**: GitHub release page
- **doxygen**: SourceForge release page
- **ClangBuildAnalyzer**: GitHub release (compute from download)
- **GDB**: GNU announcement mailing list or FTP .sig files

## Standalone Script Usage

The scripts can also be used outside Docker for local development:

```bash
# Install to /usr/local (default)
sudo ./install-ccache.sh

# Install to custom prefix
INSTALL_DIR=/opt/tools ./install-ccache.sh
```
