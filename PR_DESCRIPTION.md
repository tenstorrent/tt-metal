# Add Multi-Host Tests for BH Blitz Pipeline, Triple Pod 16x8, and 32x4 Quad Galaxy Cluster Descriptors

## Summary
This PR adds comprehensive multi-host fabric tests for three cluster descriptor configurations:
- **BH Blitz Pipeline** cluster descriptor
- **Triple Pod 16x8 Quad BH Galaxy** cluster descriptor
- **32x4 Quad Galaxy** cluster descriptor

## Changes

### Test Additions (`test_multi_host.cpp`)
Added 6 new test cases following the existing test pattern:

**BH Blitz Pipeline Tests:**
- `TestBHBlitzPipelineControlPlaneInit` - Control plane initialization test
- `TestBHBlitzPipelineFabric1DSanity` - 1D fabric connectivity validation
- `TestBHBlitzPipelineFabric2DSanity` - 2D fabric connectivity validation

**Triple Pod 16x8 Quad BH Galaxy Tests:**
- `TestTriplePod16x8QuadBHGalaxyControlPlaneInit` - Control plane initialization test
- `TestTriplePod16x8QuadBHGalaxyFabric1DSanity` - 1D fabric connectivity validation
- `TestTriplePod16x8QuadBHGalaxyFabric2DSanity` - 2D fabric connectivity validation

**Host Topology Debugging:**
- Added `print_host_topology()` helper function to log host connectivity information
- Integrated host topology printing into all ControlPlaneInit tests for better debugging

### Configuration Files
- **Created**: `tests/tt_metal/distributed/config/triple_16x8_quad_bh_galaxy_rank_bindings.yaml`
  - Rank binding configuration for 12 hosts (ranks 0-11) mapped to 3 meshes with 4 hosts each

### CI/CD Updates (`.github/workflows/fabric-cpu-only-tests-impl.yaml`)
Added test execution entries for all three cluster descriptors:

**BH Blitz Pipeline:**
- Added 5 test runs (system health, physical discovery, ControlPlaneInit, Fabric1D, Fabric2D)
- Included `--oversubscribe` flag for MPI to handle large number of ranks

**Triple Pod 16x8 Quad BH Galaxy:**
- Added 5 test runs with proper mock cluster and rank binding configuration

**32x4 Quad Galaxy:**
- Added 5 test runs leveraging existing test cases (`Test32x4QuadGalaxyControlPlaneInit`, etc.)

## Test Coverage
Each cluster descriptor now has:
- ✅ Control plane initialization validation
- ✅ 1D fabric connectivity sanity checks
- ✅ 2D fabric connectivity sanity checks
- ✅ System health reporting
- ✅ Physical discovery validation

## Testing
All tests follow the established pattern:
- Use mock cluster descriptors for CPU-only testing
- Validate intermesh connections
- Verify forwarding directions and ethernet channels
- Run in CI via `fabric-cpu-only-tests-impl.yaml` workflow
