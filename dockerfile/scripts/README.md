# Docker Build Scripts

This directory contains installation scripts for external tools used in the TT-Metalium Docker images.

## Architecture

The Docker build system uses **pre-built tool images** stored in GHCR (GitHub Container Registry) to avoid hitting third-party endpoints repeatedly. This significantly speeds up builds and reduces external dependencies.

### How It Works

1. **Tool images are built once** and pushed to GHCR by `build-docker-tools.yaml`
2. **The main Dockerfile pulls** these pre-built tool images instead of building from scratch
3. **Tool images are versioned** by `<version>-<script_hash>` to ensure cache invalidation when needed

### Layer Types

1. **Pre-built Tool Images** (9 tools)
   - Built by `Dockerfile.tools` and pushed to GHCR
   - Pulled via `FROM ${TOOL_*_IMAGE}` ARG pattern
   - Tagged as `ghcr.io/<repo>/tt-metalium/tools/<tool>:<version>-<hash>`
   - Third-party endpoints only hit once per tool version

   | Tool | Purpose |
   |------|---------|
   | ccache | Compiler cache for faster rebuilds |
   | mold | Fast linker |
   | doxygen | Documentation generator |
   | cba | ClangBuildAnalyzer for build profiling |
   | gdb | GNU Debugger |
   | cmake | Build system generator |
   | yq | YAML processor |
   | sfpi | SFPI compiler tools |
   | openmpi | MPI implementation for distributed computing |

2. **Official Image Layers** (`uv-layer`)
   - Uses `COPY --from=` with SHA256-pinned official images
   - Zero build overhead - just copies pre-built binaries
   - See: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv

### Benefits

- **No repeated downloads**: Third-party tools downloaded once, then pulled from GHCR
- **Faster CI builds**: Tool images cached in GHCR, not rebuilt every time
- **Reduced external dependencies**: Less reliance on GitHub releases, doxygen.nl, GNU mirrors
- **Hash validation**: All downloads verified with SHA256 before use
- **Reproducible builds**: Version + script hash ensures exact same binaries

### Building Tool Images

Tool images are automatically built and pushed by `build-docker-tools.yaml` when they don't exist in GHCR.

**Using build-local.sh (recommended):**
```bash
# Build dev image - automatically builds any missing tool images first
./dockerfile/build-local.sh dev
```

**Manual build:**
```bash
# Build individual tool images
docker build -f dockerfile/Dockerfile.tools --target ccache -t tool-ccache:local .
docker build -f dockerfile/Dockerfile.tools --target mold -t tool-mold:local .
docker build -f dockerfile/Dockerfile.tools --target doxygen -t tool-doxygen:local .
docker build -f dockerfile/Dockerfile.tools --target cba -t tool-cba:local .
docker build -f dockerfile/Dockerfile.tools --target gdb -t tool-gdb:local .
docker build -f dockerfile/Dockerfile.tools --target cmake -t tool-cmake:local .
docker build -f dockerfile/Dockerfile.tools --target yq -t tool-yq:local .
docker build -f dockerfile/Dockerfile.tools --target sfpi -t tool-sfpi:local .
docker build -f dockerfile/Dockerfile.tools --target openmpi -t tool-openmpi:local .

# Build main image with local tool images
docker build \
  --build-arg TOOL_CCACHE_IMAGE=tool-ccache:local \
  --build-arg TOOL_MOLD_IMAGE=tool-mold:local \
  --build-arg TOOL_DOXYGEN_IMAGE=tool-doxygen:local \
  --build-arg TOOL_CBA_IMAGE=tool-cba:local \
  --build-arg TOOL_GDB_IMAGE=tool-gdb:local \
  --build-arg TOOL_CMAKE_IMAGE=tool-cmake:local \
  --build-arg TOOL_YQ_IMAGE=tool-yq:local \
  --build-arg TOOL_SFPI_IMAGE=tool-sfpi:local \
  --build-arg TOOL_OPENMPI_IMAGE=tool-openmpi:local \
  -f dockerfile/Dockerfile --target dev .
```

## Scripts

| Script | Purpose |
|--------|---------|
| `install-ccache.sh` | Install ccache binary release |
| `install-cmake.sh` | Install CMake binary release |
| `install-clangbuildanalyzer.sh` | Build and install ClangBuildAnalyzer |
| `install-doxygen.sh` | Build and install doxygen from source |
| `install-gdb.sh` | Build and install GDB from source |
| `install-mold.sh` | Install mold linker binary release |
| `install-openmpi.sh` | Build and install OpenMPI with ULFM support |
| `install-sfpi.sh` | Install SFPI compiler tools |
| `install-yq.sh` | Install yq binary release |
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
2. Ensure compute-hashes.sh defaults match Dockerfile.tools ARGs before running
3. Run `compute-hashes.sh` to get the new SHA256 hash
4. Update the corresponding SHA256 ARG in the Dockerfile
5. Test the build

```bash
# Compute hashes for current versions
./dockerfile/scripts/compute-hashes.sh

# Compute hash for a specific version
CCACHE_VERSION=4.11.0 ./dockerfile/scripts/compute-hashes.sh
```

## Hash Verification Sources

| Tool | Source |
|------|--------|
| uv | GitHub Container Registry page (SHA256 digest shown for each version) |
| ccache | GPG signatures available at release page |
| cmake | Official CMake download page checksums |
| mold | GitHub release page |
| doxygen | SourceForge release page |
| cba | GitHub release (compute from download) |
| gdb | GNU announcement mailing list or FTP .sig files |
| yq | GitHub release checksums file |
| sfpi | Internal release artifacts |
| openmpi | OpenMPI download page checksums |

## Standalone Script Usage

The scripts can also be used outside Docker for local development:

```bash
# Install to /usr/local (default)
sudo ./install-ccache.sh

# Install to custom prefix
INSTALL_DIR=/opt/tools ./install-ccache.sh
```
