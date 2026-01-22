# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import ttnn
from models.demos.resblock.op import FusedResblock


class TestPartitionCoresByNoc:
    """Unit tests for FusedResblock.partition_cores_by_noc"""

    def test_single_core_all_noc0(self):
        """Test single core assigned to NOC0 when NOC0 has shorter hop distance"""
        # Setup
        all_matmul_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        gather_destination_core = ttnn.CoreCoord(1, 1)
        device = Mock()

        def get_hop_distance(core, dest, noc):
            if core.x == 0 and core.y == 0:
                return 2 if noc == ttnn.NOC.NOC_0 else 3
            raise ValueError(f"Unexpected core: {core}")

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        assert len(noc0_cores) == 1
        assert len(noc1_cores) == 0
        assert noc0_cores[0].x == 0 and noc0_cores[0].y == 0
        assert not noc0_core_range_set.empty()
        assert noc1_core_range_set.empty()

    def test_single_core_all_noc1(self):
        """Test single core assigned to NOC1 when NOC1 has shorter hop distance"""
        # Setup
        all_matmul_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        gather_destination_core = ttnn.CoreCoord(1, 1)
        device = Mock()

        def get_hop_distance(core, dest, noc):
            if core.x == 0 and core.y == 0:
                return 3 if noc == ttnn.NOC.NOC_0 else 2
            raise ValueError(f"Unexpected core: {core}")

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        assert len(noc0_cores) == 0
        assert len(noc1_cores) == 1
        assert noc1_cores[0].x == 0 and noc1_cores[0].y == 0
        assert noc0_core_range_set.empty()
        assert not noc1_core_range_set.empty()

    def test_tie_breaker_noc0(self):
        """Test that when hop distances are equal, NOC0 is preferred (noc0_hop <= noc1_hop)"""
        # Setup
        all_matmul_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        gather_destination_core = ttnn.CoreCoord(1, 1)
        device = Mock()

        def get_hop_distance(core, dest, noc):
            if core.x == 0 and core.y == 0:
                return 2  # Same distance for both NOCs
            raise ValueError(f"Unexpected core: {core}")

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        assert len(noc0_cores) == 1
        assert len(noc1_cores) == 0
        assert noc0_cores[0].x == 0 and noc0_cores[0].y == 0

    def test_multiple_cores_split(self):
        """Test multiple cores split between NOC0 and NOC1"""
        # Setup: 4 cores in a 2x2 grid
        all_matmul_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
            }
        )
        gather_destination_core = ttnn.CoreCoord(2, 2)
        device = Mock()

        # Mock hop distances: cores (0,0) and (0,1) prefer NOC0, cores (1,0) and (1,1) prefer NOC1
        def get_hop_distance(core, dest, noc):
            if core.x == 0 and core.y == 0:
                return 2 if noc == ttnn.NOC.NOC_0 else 3
            elif core.x == 0 and core.y == 1:
                return 2 if noc == ttnn.NOC.NOC_0 else 3
            elif core.x == 1 and core.y == 0:
                return 3 if noc == ttnn.NOC.NOC_0 else 2
            elif core.x == 1 and core.y == 1:
                return 3 if noc == ttnn.NOC.NOC_0 else 2
            else:
                raise ValueError(f"Unexpected core: {core}")

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        assert len(noc0_cores) == 2
        assert len(noc1_cores) == 2
        assert all_matmul_cores.num_cores() == 4
        # Check that cores are in the correct groups by comparing coordinates
        noc0_coords = {(core.x, core.y) for core in noc0_cores}
        noc1_coords = {(core.x, core.y) for core in noc1_cores}
        assert (0, 0) in noc0_coords
        assert (0, 1) in noc0_coords
        assert (1, 0) in noc1_coords
        assert (1, 1) in noc1_coords
        assert not noc0_core_range_set.empty()
        assert not noc1_core_range_set.empty()

    def test_all_cores_assigned(self):
        """Test that all cores are assigned to exactly one NOC group"""
        # Setup: 8 cores in a 2x4 grid
        all_matmul_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3)),
            }
        )
        gather_destination_core = ttnn.CoreCoord(2, 2)
        device = Mock()

        # Mock: alternate cores between NOC0 and NOC1
        def get_hop_distance(core, dest, noc):
            # Even x-coords prefer NOC0, odd x-coords prefer NOC1
            if core.x % 2 == 0:
                return 1 if noc == ttnn.NOC.NOC_0 else 2
            else:
                return 2 if noc == ttnn.NOC.NOC_0 else 1

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        total_assigned = len(noc0_cores) + len(noc1_cores)
        assert total_assigned == all_matmul_cores.num_cores()
        assert len(noc0_cores) == 4  # Even x-coords: (0,0), (0,1), (0,2), (0,3)
        assert len(noc1_cores) == 4  # Odd x-coords: (1,0), (1,1), (1,2), (1,3)

    def test_empty_corerangeset(self):
        """Test that empty CoreRangeSet is handled correctly"""
        # Setup
        all_matmul_cores = ttnn.CoreRangeSet([])
        gather_destination_core = ttnn.CoreCoord(1, 1)
        device = Mock()
        device.get_worker_noc_hop_distance = Mock(return_value=0)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify
        assert len(noc0_cores) == 0
        assert len(noc1_cores) == 0
        assert noc0_core_range_set.empty()
        assert noc1_core_range_set.empty()

    def test_corerange_set_correctness(self):
        """Test that CoreRangeSets contain the correct cores"""
        # Setup: 3 cores
        all_matmul_cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0)),
            }
        )
        gather_destination_core = ttnn.CoreCoord(1, 1)
        device = Mock()

        # Mock: first two cores prefer NOC0, last prefers NOC1
        def get_hop_distance(core, dest, noc):
            if core.x <= 1:
                return 1 if noc == ttnn.NOC.NOC_0 else 2
            else:
                return 2 if noc == ttnn.NOC.NOC_0 else 1

        device.get_worker_noc_hop_distance = Mock(side_effect=get_hop_distance)

        # Execute
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device
        )

        # Verify CoreRangeSets contain the correct cores
        assert noc0_core_range_set.contains(ttnn.CoreCoord(0, 0))
        assert noc0_core_range_set.contains(ttnn.CoreCoord(1, 0))
        assert not noc0_core_range_set.contains(ttnn.CoreCoord(2, 0))
        assert noc1_core_range_set.contains(ttnn.CoreCoord(2, 0))
        assert not noc1_core_range_set.contains(ttnn.CoreCoord(0, 0))
        assert not noc1_core_range_set.contains(ttnn.CoreCoord(1, 0))
