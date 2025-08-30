#!/usr/bin/env bash
set -euo pipefail

# Minimal runner: assumes the unit_tests_api binary exists and is runnable.
# Pass the binary path as the first argument or set UNIT_TESTS_API_BIN.
# Example:
#   ./scripts/run_allocator_tests.sh /path/to/build/test/tt_metal/unit_tests_api

BIN="${1:-${UNIT_TESTS_API_BIN:-unit_tests_api}}"

# Exact list of allocator-related GoogleTests to run
GTESTS=(
  # FreeListOpt allocator algorithm
  'FreeListOptTest.Allocation'
  'FreeListOptTest.Alignment'
  'FreeListOptTest.MinAllocationSize'
  'FreeListOptTest.Clear'
  'FreeListOptTest.AllocationAndDeallocation'
  'FreeListOptTest.AllocateAtAddress'
  'FreeListOptTest.AllocateAtAddressInteractions'
  'FreeListOptTest.ShrinkAndReset'
  'FreeListOptTest.Statistics'
  'FreeListOptTest.AllocateFromTop'
  'FreeListOptTest.Coalescing'
  'FreeListOptTest.CoalescingAfterResetShrink'
  'FreeListOptTest.OutOfMemory'
  'FreeListOptTest.AvailableAddresses'
  'FreeListOptTest.LowestOccupiedAddress'
  'FreeListOptTest.LowestOccupiedAddressWithAllocateAt'
  'FreeListOptTest.FirstFit'
  'FreeListOptTest.FirstFitAllocateAtAddressInteractions'

  # L1 banking allocator behavior
  'DeviceSingleCardBufferFixture.TestL1BuffersAllocatedTopDown'
  'DeviceSingleCardBufferFixture.TestL1BuffersDoNotGrowBeyondBankSize'

  # Banked buffer tests (exercise allocator/banking paths)
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedL1ReaderOnly'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1ReaderOnly'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedDramReaderOnly'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramReaderOnly'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedL1WriterOnly'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1WriterOnly'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedDramWriterOnly'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramWriterOnly'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedL1ReaderAndWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1ReaderAndWriter'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedDramReaderAndWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramReaderAndWriter'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedDramReaderAndL1Writer'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramReaderAndL1Writer'
  'MeshDeviceFixture.TensixTestSingleCoreSingleTileBankedL1ReaderAndDramWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1ReaderAndDramWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1ReaderDataCopyL1Writer'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramReaderDataCopyDramWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedL1ReaderDataCopyDramWriter'
  'MeshDeviceFixture.TensixTestSingleCoreMultiTileBankedDramReaderDataCopyL1Writer'
)

echo "Running ${#GTESTS[@]} tests with: $BIN"
for t in "${GTESTS[@]}"; do
  echo "==> $t"
  "$BIN" --gtest_color=yes --gtest_filter="$t"
done
