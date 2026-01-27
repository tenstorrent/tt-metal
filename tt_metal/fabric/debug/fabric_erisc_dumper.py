#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ERISC Value Dumper - Fabric Debugging Tool

This tool reads values from specific addresses in ERISC (Ethernet RISC) cores across
Tenstorrent devices using the ttexalens library. It's specifically designed for debugging
the fabric EDM (Ethernet Data Movement) system.

Features:
- Single snapshot mode: Read register values once
- Polling mode: Monitor register changes over time with delta analysis
- Buffer mode: Read and analyze circular buffer contents
- Fabric stream register mode: Display fabric flow control registers in matrix format
- Multi-device and selective core filtering support
- JSON, CSV, and console output formats

Usage Examples:
    # Basic snapshot of fabric registers
    python dump_erisc_values.py

    # Monitor fabric stream registers for 10 seconds
    python dump_erisc_values.py --fabric-streams --poll --duration 10

    # Check specific addresses on device 0, cores (0,5) and (0,7)
    python dump_erisc_values.py --addresses 0xFFB121B0,0xFFB121F0 --devices 0 --cores 0,5,0,7

    # Analyze buffer contents in 4 slots of 64 bytes each
    python dump_erisc_values.py --buffer-mode --addresses 0x12345678 --num-elements 4

    # Export polling data to CSV for analysis
    python dump_erisc_values.py --poll --csv --output fabric_debug.csv --duration 30

For detailed option descriptions, use: python dump_erisc_values.py --help
"""

import argparse
import sys
from typing import List, Dict, Optional, Tuple
import json
import time
import csv
from datetime import datetime

# ttexalens imports
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device

from fabric_erisc_constants import (
    ERISC_REGISTERS,
    DEFAULT_ADDRESSES,
    FABRIC_STREAM_GROUPS,
    STREAM_REGISTER_MASK_BITS,
    STREAM_REGISTER_MASK,
)
from fabric_erisc_utils import (
    get_stream_reg_address,
    normalize_architecture,
    get_fabric_stream_addresses,
    detect_device_architecture,
    parse_core_key,
)


class ERISCDumper:
    """
    ERISC Value Dumper for fabric debugging.

    This class provides methods to read values from ERISC (Ethernet RISC) cores
    across Tenstorrent devices, with support for multiple operational modes
    including single snapshots, continuous polling, buffer analysis, and
    fabric stream register monitoring.
    """

    def __init__(self):
        """Initialize the ERISC dumper with ttexalens context."""
        self.context = init_ttexalens()
        print("Initialized ttexalens context")

    def format_value(self, value: int, use_decimal: bool = False) -> str:
        """Format a value as either hex or decimal with consistent width."""
        if use_decimal:
            return f"{value:10,d}"
        else:
            return f"0x{value:08x}"

    def get_devices(self) -> List:
        """Get all available devices from the context."""
        devices = []
        try:
            # Get all devices from context
            for device_id, device in self.context.devices.items():
                devices.append(device)
            print(f"Found {len(devices)} device(s)")
        except Exception as e:
            print(f"Error getting devices: {e}")
        return devices

    def get_filtered_devices(self, device_ids: List[int] = None) -> List:
        """Get devices filtered by device IDs if specified."""
        all_devices = self.get_devices()

        if device_ids is None:
            return all_devices

        filtered_devices = []
        available_device_ids = [device.id for device in all_devices]

        for device in all_devices:
            if device.id in device_ids:
                filtered_devices.append(device)

        # Report which requested devices were not found
        missing_devices = set(device_ids) - set(available_device_ids)
        if missing_devices:
            print(f"Warning: Requested device IDs not found: {sorted(missing_devices)}")

        if not filtered_devices:
            if device_ids:
                print(f"No devices found matching IDs: {device_ids}")
                print(f"Available devices: {available_device_ids}")
            return []

        print(f"Filtering to devices: {[dev._id for dev in filtered_devices]}")
        return filtered_devices

    def get_ethernet_cores(self, device, filter_active=True, check_reset=True, core_filter_coords=None):
        """Get ethernet (ERISC) core locations for a device with optional filtering.

        Args:
            device: The ttexalens device to get cores from
            filter_active: Only include active cores (default: True)
            check_reset: Check if cores are out of reset (default: True)
            core_filter_coords: List of (x,y) tuples to filter cores by (default: None for all)

        Returns:
            List of OnChipCoordinate objects for accessible ethernet cores
        """
        try:
            # Get ethernet core locations using the proper ttexalens API (seen in other debugging scripts)
            eth_locations = device.get_block_locations("eth")

            if not eth_locations:
                print(f"Warning: No ethernet cores found for device {device.id}")
                return []

            filtered_locations = []
            coord_strs = []

            for loc in eth_locations:
                coord_info = loc.to("logical")
                if isinstance(coord_info, tuple) and len(coord_info) >= 2:
                    if isinstance(coord_info[0], tuple) and len(coord_info[0]) == 2:
                        x, y = coord_info[0]
                    else:
                        continue
                else:
                    continue

                # Get the NOC block for this location
                try:
                    noc_block = device.get_block(loc)
                    status_info = []

                    # Check if it's active or idle (pattern from dump_callstacks.py)
                    if filter_active and hasattr(device, "idle_eth_blocks"):
                        if noc_block in device.idle_eth_blocks:
                            status_info.append("idle")
                        else:
                            status_info.append("active")

                        # Skip idle cores if we only want active ones
                        if filter_active and noc_block in device.idle_eth_blocks:
                            continue

                    # Check reset status if requested
                    if check_reset:
                        try:
                            # Try to read a reset-related register to see if core is responsive
                            # ETH_RISC_RESET register at offset 0x21B0 from ETH_RISC_REGS_START
                            reset_status = self.read_address(device, loc, 0xFFB121B0)
                            if reset_status is not None:
                                if reset_status != 0:
                                    status_info.append("not_reset")
                                else:
                                    status_info.append("in_reset")
                                    continue  # Skip cores that are in reset
                            else:
                                status_info.append("unreadable")
                                continue  # Skip cores we can't read from
                        except:
                            status_info.append("read_error")
                            continue

                    # Check core coordinate filter
                    if core_filter_coords is not None:
                        if (x, y) not in core_filter_coords:
                            continue  # Skip cores not in the filter list

                    # Add to filtered list
                    filtered_locations.append(loc)
                    status_str = f"({x},{y})" + (f"[{','.join(status_info)}]" if status_info else "")
                    coord_strs.append(status_str)

                except Exception as e:
                    print(f"  Warning: Error checking core ({x},{y}): {e}")
                    continue

            if not filtered_locations:
                print(f"  No accessible ethernet cores found (after filtering)")
                return []
            else:
                # Don't print here - let caller handle display
                return filtered_locations

        except Exception as e:
            print(f"Error getting ethernet cores for device {device.id}: {e}")
            return []

    def read_address(self, device, coord: OnChipCoordinate, address: int) -> Optional[int]:
        """Read a 32-bit value from a specific address on an ethernet core.

        Args:
            device: The ttexalens device
            coord: OnChipCoordinate for the target ethernet core
            address: Memory address to read from

        Returns:
            32-bit integer value or None if read failed
        """
        try:
            # Read 4 bytes (32-bit value) from the specific address using ttexalens API
            read_data = read_from_device(coord, address, device.id, 4, self.context)

            # Convert bytes to 32-bit integer (little endian)
            if isinstance(read_data, bytes) and len(read_data) >= 4:
                return int.from_bytes(read_data[:4], byteorder="little")
            elif isinstance(read_data, (list, tuple)) and len(read_data) >= 4:
                # If it's a list of bytes, convert to integer
                return read_data[0] | (read_data[1] << 8) | (read_data[2] << 16) | (read_data[3] << 24)
            elif isinstance(read_data, int):
                return read_data
            else:
                coord_str = coord.to_user_str() if hasattr(coord, "to_user_str") else str(coord)
                print(f"Warning: Unexpected data type from device {device.id} core {coord_str}: {type(read_data)}")
                return None

        except Exception as e:
            coord_str = coord.to_user_str() if hasattr(coord, "to_user_str") else str(coord)
            print(f"Error reading address 0x{address:08x} from device {device.id} core {coord_str}: {e}")
            return None

    def dump_values(
        self,
        addresses: List[int],
        filter_active: bool = True,
        check_reset: bool = True,
        buffer_mode: bool = False,
        num_elements: int = 4,
        slot_size: int = 64,
        data_per_slot: int = 16,
        use_decimal: bool = False,
        device_filter_ids: List[int] = None,
        core_filter_coords: List[Tuple[int, int]] = None,
    ) -> Dict:
        """Dump values from specified addresses across filtered devices and ethernet cores."""
        results = {}
        devices = self.get_filtered_devices(device_filter_ids)

        if not devices:
            print("No devices found!")
            return results

        # Track whether buffer column headers have been printed
        buffer_headers_printed = False
        current_buffer_address = None

        for device in devices:
            device_id = device.id
            results[device_id] = {}

            eth_cores = self.get_ethernet_cores(
                device, filter_active=filter_active, check_reset=check_reset, core_filter_coords=core_filter_coords
            )

            if not eth_cores:
                print(f"Device {device_id}: No ethernet cores found")
                continue

            for core_idx, core_coord in enumerate(eth_cores):
                is_last_core = core_idx == len(eth_cores) - 1

                # Extract (x, y) coordinates for the key and display
                coord_info = core_coord.to("logical")
                if isinstance(coord_info, tuple) and len(coord_info) >= 2:
                    if isinstance(coord_info[0], tuple) and len(coord_info[0]) == 2:
                        core_x, core_y = coord_info[0]
                    else:
                        print(f"  Warning: Unexpected coordinate format: {coord_info}")
                        continue
                else:
                    print(f"  Warning: Unexpected coordinate format: {coord_info}")
                    continue

                core_key = f"core_{core_x}_{core_y}"
                results[device_id][core_key] = {}

                if buffer_mode:
                    # Buffer mode - read and display buffers
                    for address in addresses:
                        slots = self.read_buffer(device, core_coord, address, num_elements, slot_size, data_per_slot)
                        results[device_id][core_key][f"0x{address:08x}"] = slots

                        if slots is not None:
                            # Check if we need to print a new buffer address header
                            print_buffer_address = current_buffer_address != address
                            if print_buffer_address:
                                current_buffer_address = address

                            self.print_buffer(
                                device_id,
                                core_x,
                                core_y,
                                address,
                                slots,
                                data_per_slot,
                                use_decimal,
                                buffer_headers_printed,
                                print_buffer_address,
                                is_last_core,
                            )
                            buffer_headers_printed = True  # Mark headers as printed after first buffer
                        else:
                            print(f"  Dev{device_id} Core({core_x},{core_y}): Buffer 0x{address:08x}: <read failed>")
                else:
                    # Normal mode - read individual addresses
                    for address in addresses:
                        value = self.read_address(device, core_coord, address)
                        results[device_id][core_key][f"0x{address:08x}"] = value

                        if value is not None:
                            formatted_addr = f"0x{address:08x}"
                            formatted_value = self.format_value(value, use_decimal)
                            decimal_part = f" ({value:,d})" if not use_decimal else ""
                            core_label = f"Dev{device_id} Core({core_x},{core_y}):"
                            print(f"  {core_label:<17s} {formatted_addr}: {formatted_value}{decimal_part}")
                        else:
                            core_label = f"Dev{device_id} Core({core_x},{core_y}):"
                            print(f"  {core_label:<17s} 0x{address:08x}: <read failed>")

            # Print separator after each device (except for buffer mode which handles it separately)
            if not buffer_mode and is_last_core:
                print("-" * 80)

        return results

    def poll_values(
        self,
        addresses: List[int],
        filter_active: bool = True,
        check_reset: bool = True,
        interval: float = 0.1,
        duration: float = 10.0,
        changes_only: bool = False,
        csv_output: bool = False,
        output_file: Optional[str] = None,
        show_delta_summary: bool = True,
        use_decimal: bool = False,
        device_filter_ids: List[int] = None,
        core_filter_coords: List[Tuple[int, int]] = None,
    ) -> List[Dict]:
        """Poll values from ERISC cores at regular intervals."""
        devices = self.get_filtered_devices(device_filter_ids)

        if not devices:
            print("No devices found!")
            return []

        # Build list of all device/core combinations to monitor
        polling_targets = []
        for device in devices:
            device_id = device.id
            eth_cores = self.get_ethernet_cores(
                device, filter_active=filter_active, check_reset=check_reset, core_filter_coords=core_filter_coords
            )

            for core_coord in eth_cores:
                # Extract (x, y) coordinates for display
                coord_info = core_coord.to("logical")
                if isinstance(coord_info, tuple) and len(coord_info) >= 2:
                    if isinstance(coord_info[0], tuple) and len(coord_info[0]) == 2:
                        core_x, core_y = coord_info[0]
                        polling_targets.append(
                            {
                                "device_id": device_id,
                                "device": device,
                                "coord": core_coord,
                                "core_x": core_x,
                                "core_y": core_y,
                                "core_key": f"dev{device_id}_core_{core_x}_{core_y}",
                            }
                        )

        if not polling_targets:
            print("No cores available for polling!")
            return []

        print(f"\nPolling {len(polling_targets)} core(s) across {len(devices)} device(s)")
        print(f"Interval: {interval*1000:.1f}ms, Duration: {duration:.1f}s, Samples: ~{int(duration/interval)}")
        print(f"Addresses: {[f'0x{addr:08x}' for addr in addresses]}")

        # Initialize CSV writer if needed
        csv_writer = None
        csv_file = None
        if csv_output:
            if output_file:
                csv_file = open(output_file, "w", newline="")
            else:
                csv_file = sys.stdout

            fieldnames = ["timestamp", "elapsed_ms"] + [
                f'{target["core_key"]}_0x{addr:08x}' for target in polling_targets for addr in addresses
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        # Store previous values for change detection
        previous_values = {}
        polling_results = []
        start_time = time.time()
        sample_count = 0

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time

                if elapsed >= duration:
                    break

                sample_count += 1
                timestamp = datetime.now()
                elapsed_ms = int(elapsed * 1000)

                # Read all values for this sample
                sample_data = {
                    "timestamp": timestamp.isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "sample": sample_count,
                    "values": {},
                }

                csv_row = {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "elapsed_ms": elapsed_ms}

                changed_this_sample = False

                for target in polling_targets:
                    device = target["device"]
                    coord = target["coord"]
                    core_key = target["core_key"]
                    core_x, core_y = target["core_x"], target["core_y"]

                    sample_data["values"][core_key] = {}

                    for address in addresses:
                        addr_key = f"0x{address:08x}"
                        value = self.read_address(device, coord, address)

                        sample_data["values"][core_key][addr_key] = value
                        csv_col_key = f"{core_key}_{addr_key}"
                        csv_row[csv_col_key] = value if value is not None else "ERROR"

                        # Track changes
                        prev_key = f"{core_key}_{addr_key}"
                        if prev_key in previous_values:
                            if previous_values[prev_key] != value:
                                changed_this_sample = True
                        else:
                            changed_this_sample = True  # First sample always counts as changed
                        previous_values[prev_key] = value

                # Output this sample
                if not changes_only or changed_this_sample:
                    if csv_output:
                        csv_writer.writerow(csv_row)
                        if csv_file != sys.stdout:
                            csv_file.flush()  # Ensure data is written immediately
                    else:
                        # Console output
                        print(f"\n[{timestamp.strftime('%H:%M:%S.%f')[:-3]}] Sample {sample_count} (+{elapsed_ms}ms):")
                        for target in polling_targets:
                            core_key = target["core_key"]
                            core_x, core_y = target["core_x"], target["core_y"]

                            values_str = []
                            for address in addresses:
                                addr_key = f"0x{address:08x}"
                                value = sample_data["values"][core_key][addr_key]
                                if value is not None:
                                    formatted_value = self.format_value(value, use_decimal)
                                    values_str.append(f"{addr_key}={formatted_value}")
                                else:
                                    values_str.append(f"{addr_key}=ERROR")

                            core_label = f"Dev{target['device_id']} Core({core_x},{core_y}):"
                            print(f"  {core_label:<17s} {', '.join(values_str)}")

                polling_results.append(sample_data)

                # Sleep until next interval
                next_time = start_time + (sample_count * interval)
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n\nPolling interrupted by user after {elapsed:.1f}s ({sample_count} samples)")

        finally:
            if csv_file and csv_file != sys.stdout:
                csv_file.close()

        total_elapsed = time.time() - start_time
        print(f"\nPolling completed: {sample_count} samples over {total_elapsed:.1f}s")
        print(f"Average interval: {total_elapsed/sample_count*1000:.1f}ms")

        # Generate delta summary
        if show_delta_summary and len(polling_results) >= 2:
            self._print_delta_summary(polling_results, total_elapsed, polling_targets, addresses, use_decimal)

        return polling_results

    def read_fabric_stream_registers(
        self,
        addresses: List[int] = None,
        group_name: str = None,
        filter_active: bool = True,
        check_reset: bool = True,
        device_filter_ids: List[int] = None,
        core_filter_coords: List[Tuple[int, int]] = None,
    ) -> Dict:
        """Read fabric stream registers from filtered devices and ethernet cores."""
        devices = self.get_filtered_devices(device_filter_ids)

        if not devices:
            print("No devices found!")
            return {}

        # Use default fabric streams if no addresses specified
        if addresses is None:
            if not devices:
                return {}
            # Use first device to get architecture for address calculation
            arch = detect_device_architecture(devices[0])
            addresses_dict = get_fabric_stream_addresses(group_name, arch=arch)
            addresses = list(addresses_dict.values())

        results = {}

        for device in devices:
            device_id = device.id
            results[device_id] = {}

            # Detect architecture for masking decision
            arch = detect_device_architecture(device)
            apply_masking = arch != "wormhole"  # Apply masking on non-Wormhole (e.g., Blackhole)

            eth_cores = self.get_ethernet_cores(
                device, filter_active=filter_active, check_reset=check_reset, core_filter_coords=core_filter_coords
            )

            if not eth_cores:
                print(f"Device {device_id}: No ethernet cores found")
                continue

            for core_coord in eth_cores:
                # Extract (x, y) coordinates for the key
                coord_info = core_coord.to("logical")
                if isinstance(coord_info, tuple) and len(coord_info) >= 2:
                    if isinstance(coord_info[0], tuple) and len(coord_info[0]) == 2:
                        core_x, core_y = coord_info[0]
                    else:
                        print(f"  Warning: Unexpected coordinate format: {coord_info}")
                        continue
                else:
                    print(f"  Warning: Unexpected coordinate format: {coord_info}")
                    continue

                core_key = f"core_{core_x}_{core_y}"
                results[device_id][core_key] = {}

                for address in addresses:
                    raw_value = self.read_address(device, core_coord, address)
                    if raw_value is not None and apply_masking:
                        # Apply same masking as kernel does on non-Wormhole architectures
                        # get_ptr_val() uses: (register_value & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1))
                        value = raw_value & STREAM_REGISTER_MASK
                    else:
                        value = raw_value
                    results[device_id][core_key][f"0x{address:08x}"] = value

        return results

    def print_fabric_stream_matrix(
        self, results: Dict, group_name: str = None, arch: str = "wormhole", use_decimal: bool = False
    ):
        """Print fabric stream registers in matrix format.

        Args:
            results: Dictionary of fabric stream register results
            group_name: Fabric stream group name
            arch: Device architecture
            use_decimal: If True, display values in decimal instead of hex
        """
        if group_name is None:
            group_name = "all_fabric_free_slots"

        if group_name not in FABRIC_STREAM_GROUPS:
            print(f"Error: Unknown fabric stream group: {group_name}")
            return

        group = FABRIC_STREAM_GROUPS[group_name]

        print(f"\n=== FABRIC STREAM REGISTERS ===")
        print(f"{group['title']}:")
        print(f"({group['description']})")

        # Build column headers and show addresses
        headers = []
        addresses = []
        for i, stream_id in enumerate(group["stream_ids"]):
            headers.append(f"Stream{stream_id}")
            try:
                addr = get_stream_reg_address(stream_id, "BUF_SPACE_AVAILABLE", arch)
                addresses.append(f"0x{addr:08x}")
            except KeyError:
                addresses.append("  <err>")

        # Print address information first
        print()
        print("Stream register addresses:")
        addr_header_line = " " * 19 + "  ".join(f"{h:>10s}" for h in headers)
        print(addr_header_line)
        addr_line = " " * 19 + "  ".join(f"{a:>10s}" for a in addresses)
        print(addr_line)

        print()
        print("Register values:")
        # Print value header row
        header_line = " " * 19 + "  ".join(f"{h:>10s}" for h in headers)
        print(header_line)

        # Print data rows with device separators
        device_ids = list(results.keys())
        for dev_idx, device_id in enumerate(device_ids):
            device_data = results[device_id]
            is_last_device = dev_idx == len(device_ids) - 1

            for core_key, core_data in device_data.items():
                # Parse core coordinates from core_key
                core_x, core_y = parse_core_key(core_key)
                row_label = f"Dev{device_id} Core({core_x},{core_y}):"

                values = []
                for stream_id in group["stream_ids"]:
                    # Calculate expected address for this stream register
                    try:
                        addr = get_stream_reg_address(stream_id, "BUF_SPACE_AVAILABLE", arch)
                        addr_key = f"0x{addr:08x}"
                        value = core_data.get(addr_key, None)
                        if value is not None:
                            formatted_value = self.format_value(value, use_decimal)
                            values.append(f"{formatted_value:>10s}")
                        else:
                            values.append("    <fail>")
                    except KeyError as e:
                        values.append("    <err>")

                row = f"{row_label:<19s}" + "  ".join(f"{v:>10s}" for v in values)
                print(row)

            # Print separator after each device (except the last one)
            if not is_last_device:
                # Calculate line width: label (19) + data columns (10 chars each) + separators (2 spaces each)
                num_streams = len(group["stream_ids"])
                # Format: "Label (19)" + "Value1 (10)" + "  " + "Value2 (10)" + "  " + ...
                # = Label + (num_streams * 10) + ((num_streams - 1) * 2)
                total_width = 19 + num_streams * 10 + (num_streams - 1) * 2
                print("-" * total_width)

    def read_buffer(
        self, device, coord: OnChipCoordinate, address: int, num_elements: int, slot_size: int, data_per_slot: int
    ) -> Optional[List[List[int]]]:
        """Read buffer data from memory and return as list of slots, each containing words."""
        try:
            total_size = num_elements * slot_size
            raw_data = read_from_device(coord, address, device.id, total_size, self.context)

            if raw_data is None:
                return None

            # Convert raw data to list of integers if needed
            if isinstance(raw_data, int):
                # Single value case - shouldn't happen with large reads
                return None
            elif isinstance(raw_data, (bytes, bytearray)):
                # Convert bytes to list of 32-bit integers
                data_ints = []
                for i in range(0, len(raw_data), 4):
                    if i + 4 <= len(raw_data):
                        # Little-endian 32-bit word
                        word = int.from_bytes(raw_data[i : i + 4], byteorder="little")
                        data_ints.append(word)
                raw_data = data_ints
            elif isinstance(raw_data, list):
                # Already a list - might be bytes or ints
                if raw_data and isinstance(raw_data[0], int):
                    # List of bytes, convert to 32-bit words
                    data_ints = []
                    for i in range(0, len(raw_data), 4):
                        if i + 3 < len(raw_data):
                            # Combine 4 bytes into 32-bit word (little-endian)
                            word = (
                                (raw_data[i + 3] << 24) | (raw_data[i + 2] << 16) | (raw_data[i + 1] << 8) | raw_data[i]
                            )
                            data_ints.append(word)
                    raw_data = data_ints

            # Parse into slots
            slots = []
            words_per_slot = data_per_slot // 4  # Number of 32-bit words per slot
            words_between_slots = slot_size // 4  # Total words in each slot (including unused data)

            for slot_idx in range(num_elements):
                slot_start_word = slot_idx * words_between_slots
                slot_words = []

                for word_idx in range(words_per_slot):
                    data_word_idx = slot_start_word + word_idx
                    if data_word_idx < len(raw_data):
                        slot_words.append(raw_data[data_word_idx])
                    else:
                        slot_words.append(0)  # Pad with zeros if not enough data

                slots.append(slot_words)

            return slots

        except Exception as e:
            print(f"Error reading buffer at 0x{address:08x}: {e}")
            return None

    def print_buffer(
        self,
        device_id: int,
        core_x: int,
        core_y: int,
        address: int,
        slots: List[List[int]],
        data_per_slot: int,
        use_decimal: bool = False,
        headers_printed: bool = False,
        print_buffer_address: bool = True,
        is_last_core: bool = False,
    ):
        """Print buffer data in matrix format with slots as columns."""
        if not slots or not slots[0]:
            print(f"Dev{device_id} Core({core_x},{core_y}) @ 0x{address:08x}: <no data>")
            return

        num_slots = len(slots)
        words_per_slot = len(slots[0])

        # Print buffer address header only when it changes
        if print_buffer_address:
            print(f"\nBuffer @ 0x{address:08x}:")

        # Use fixed width for device/core label to handle varying coordinate sizes
        # Max label is like "Dev99 Core(99,99):" = 21 chars, use 22 for safety
        fixed_label_width = 22

        # Print column headers (slot numbers) only once
        if not headers_printed:
            slot_headers = [f"Slot{i}" for i in range(num_slots)]
            # Align headers: account for fixed label width + spacing + index "[0] "
            header_indent = fixed_label_width + 4 + 4  # label + spacing + "[0] "
            header_line = " " * header_indent + "    ".join(f"{h:>10s}" for h in slot_headers)
            print(f"{header_line}")

        # Create device/core label with fixed width
        device_core_label = f"Dev{device_id} Core({core_x},{core_y}):"

        # Print each word as a row
        for word_idx in range(words_per_slot):
            word_values = []
            for slot_idx in range(num_slots):
                if word_idx < len(slots[slot_idx]):
                    value = slots[slot_idx][word_idx]
                    word_values.append(self.format_value(value, use_decimal))
                else:
                    zero_value = "         0" if use_decimal else "0x00000000"
                    word_values.append(zero_value)

            data_line = f"[{word_idx}] " + "    ".join(f"{val:>10s}" for val in word_values)

            if word_idx == 0:
                # First row: print on same line as device/core label with fixed width
                print(f"\n{device_core_label:<{fixed_label_width}}    {data_line}")
            else:
                # Subsequent rows: align with first row's data
                print(f"{' ' * (fixed_label_width + 4)}{data_line}")

        # Print separator line only after the last core of each device
        if is_last_core:
            # Calculate total width: label + spacing + index + data
            data_width = 4 + num_slots * 10 + (num_slots - 1) * 4  # "[0] " + values + separators
            total_width = fixed_label_width + 4 + data_width
            print("-" * total_width)

    def _print_delta_summary(
        self,
        polling_results: List[Dict],
        total_elapsed: float,
        polling_targets: List[Dict],
        addresses: List[int],
        use_decimal: bool = False,
    ):
        """Print a summary of value changes during polling."""
        if len(polling_results) < 2:
            return

        first_sample = polling_results[0]
        last_sample = polling_results[-1]

        print(f"\n=== DELTA SUMMARY ===")
        print(f"Time span: {first_sample['timestamp'][:19]} → {last_sample['timestamp'][:19]}")
        print(f"Duration: {total_elapsed:.2f}s")

        changes_found = False

        for target in polling_targets:
            core_key = target["core_key"]
            core_x, core_y = target["core_x"], target["core_y"]
            device_id = target["device_id"]

            core_changes = []

            for address in addresses:
                addr_key = f"0x{address:08x}"

                # Get initial and final values
                try:
                    initial_val = first_sample["values"][core_key][addr_key]
                    final_val = last_sample["values"][core_key][addr_key]

                    # Skip if either value is None (read error)
                    if initial_val is None or final_val is None:
                        continue

                    # Calculate delta
                    delta = final_val - initial_val

                    # Add to summary (including unchanged addresses)
                    change_info = {"address": addr_key, "initial": initial_val, "final": final_val, "delta": delta}
                    core_changes.append(change_info)

                except (KeyError, TypeError):
                    continue

            # Print changes for this core
            if core_changes:
                changes_found = True
                for change in core_changes:
                    addr = change["address"]
                    initial = change["initial"]
                    final = change["final"]
                    delta = change["delta"]

                    # Format the output nicely
                    delta_sign = "+" if delta > 0 else ""
                    initial_formatted = self.format_value(initial, use_decimal)
                    final_formatted = self.format_value(final, use_decimal)
                    decimal_part = f" ({initial:,d}) → ({final:,d})" if not use_decimal else ""
                    core_label = f"Dev{device_id} Core({core_x},{core_y}):"
                    print(
                        f"  {core_label:<17s} {addr}: {initial_formatted} → {final_formatted}{decimal_part} (Δ{delta_sign}{delta:,d})"
                    )

        if not changes_found:
            print("  No accessible addresses found for delta analysis")

        print()  # Extra newline for spacing


def parse_addresses(addr_string: str) -> List[int]:
    """Parse comma-separated address string into list of integers."""
    addresses = []
    for addr in addr_string.split(","):
        addr = addr.strip()
        try:
            if addr.startswith("0x"):
                addresses.append(int(addr, 16))
            else:
                addresses.append(int(addr, 10))
        except ValueError:
            print(f"Warning: Invalid address format '{addr}', skipping")
    return addresses


def parse_device_ids(device_string: str) -> List[int]:
    """Parse comma-separated device ID string into list of integers."""
    device_ids = []
    for device_id in device_string.split(","):
        device_id = device_id.strip()
        try:
            device_ids.append(int(device_id))
        except ValueError:
            print(f"Warning: Invalid device ID format '{device_id}', skipping")
    return device_ids


def parse_core_coordinates(core_string: str) -> List[Tuple[int, int]]:
    """Parse comma-separated core coordinate string into list of (x,y) tuples.

    Expected format: x1,y1,x2,y2,x3,y3,... where pairs represent (x,y) coordinates.
    Example: "0,5,0,7,1,3" -> [(0,5), (0,7), (1,3)]
    """
    coordinates = []
    parts = [part.strip() for part in core_string.split(",")]

    if len(parts) % 2 != 0:
        print(f"Warning: Core coordinates must be in x,y pairs. Got {len(parts)} values, expected even number.")
        return coordinates

    for i in range(0, len(parts), 2):
        try:
            x = int(parts[i])
            y = int(parts[i + 1])
            coordinates.append((x, y))
        except (ValueError, IndexError):
            print(
                f"Warning: Invalid core coordinate pair '{parts[i]},{parts[i+1] if i+1 < len(parts) else '?'}', skipping"
            )

    return coordinates


def main():
    parser = argparse.ArgumentParser(
        description="Dump values from ERISC cores for fabric debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic options
    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument(
        "--addresses", type=str, help="Comma-separated hex addresses to read (default: common ERISC registers)"
    )
    basic_group.add_argument("--output", type=str, help="Output file path (default: stdout)")
    basic_group.add_argument("--json", action="store_true", help="Output in JSON format")
    basic_group.add_argument(
        "--decimal", action="store_true", help="Show values in decimal instead of hex (applies to all output modes)"
    )

    # Filtering options
    filter_group = parser.add_argument_group("Core and Device Filtering")
    filter_group.add_argument(
        "--include-idle", action="store_true", help="Include idle ethernet cores (default: active only)"
    )
    filter_group.add_argument("--skip-reset-check", action="store_true", help="Skip checking if cores are out of reset")
    filter_group.add_argument(
        "--devices", type=str, help="Comma-separated list of device IDs to include (default: all devices)"
    )
    filter_group.add_argument(
        "--cores",
        type=str,
        help="Comma-separated list of core coordinates as x1,y1,x2,y2,... (e.g., 0,5,0,7 for cores (0,5) and (0,7))",
    )

    # Polling mode options
    polling_group = parser.add_argument_group("Polling Mode")
    polling_group.add_argument("--poll", action="store_true", help="Enable polling mode to monitor values over time")
    polling_group.add_argument(
        "--interval", type=float, default=0.1, help="Polling interval in seconds (default: 0.1 = 100ms)"
    )
    polling_group.add_argument(
        "--duration", type=float, default=10.0, help="Polling duration in seconds (default: 10.0)"
    )
    polling_group.add_argument("--csv", action="store_true", help="Output polling data in CSV format")
    polling_group.add_argument(
        "--changes-only", action="store_true", help="In polling mode, only show values that changed"
    )
    polling_group.add_argument(
        "--no-delta-summary", action="store_true", help="Skip the delta summary at the end of polling"
    )

    # Buffer mode options
    buffer_group = parser.add_argument_group("Buffer Mode")
    buffer_group.add_argument(
        "--buffer-mode", action="store_true", help="Enable buffer mode - treat addresses as buffer starts"
    )
    buffer_group.add_argument(
        "--num-elements", type=int, default=4, help="Number of elements in each buffer (default: 4)"
    )
    buffer_group.add_argument(
        "--slot-size", type=int, default=64, help="Size of each buffer slot in bytes (default: 64)"
    )
    buffer_group.add_argument(
        "--data-per-slot", type=int, default=16, help="Amount of data to read per slot in bytes (default: 16)"
    )

    # Fabric stream register options
    fabric_group = parser.add_argument_group("Fabric Stream Register Mode")
    fabric_group.add_argument(
        "--fabric-streams", action="store_true", help="Enable fabric stream register dumping mode"
    )
    fabric_group.add_argument(
        "--stream-group",
        choices=[
            # Buffer free slots (flow control) - default
            "sender_free_slots",
            "receiver_free_slots",
            "all_fabric_free_slots",
            # Ack/completion streams (remote buffer status)
            "sender_acks",
            "sender_completions",
            "receiver_pkts_sent",
            "all_acks_and_completions",
        ],
        default="all_fabric_free_slots",
        help="Fabric stream group to dump (default: all_fabric_free_slots). Note: ack/completion streams show REMOTE buffer status - low/zero values are normal when idle",
    )

    args = parser.parse_args()

    # Validate mutually exclusive modes
    mode_count = sum([args.poll, args.buffer_mode, args.fabric_streams])
    if mode_count > 1:
        print("Error: Only one mode can be used at a time: --poll, --buffer-mode, or --fabric-streams")
        sys.exit(1)

    # Validate buffer mode arguments
    if args.buffer_mode:
        if args.data_per_slot > args.slot_size:
            print("Error: data-per-slot cannot be larger than slot-size")
            sys.exit(1)
        if args.data_per_slot % 4 != 0:
            print("Error: data-per-slot must be a multiple of 4 (word-aligned)")
            sys.exit(1)
        if args.slot_size % 4 != 0:
            print("Error: slot-size must be a multiple of 4 (word-aligned)")
            sys.exit(1)
        if args.num_elements < 1:
            print("Error: num-elements must be at least 1")
            sys.exit(1)

    # Validate fabric streams mode
    if args.fabric_streams:
        if args.addresses:
            print(
                "Error: --addresses cannot be used with --fabric-streams (stream addresses are calculated automatically)"
            )
            sys.exit(1)

    # Parse addresses (not used for fabric streams mode)
    if args.fabric_streams:
        addresses = None  # Will be calculated automatically
    elif args.addresses:
        addresses = parse_addresses(args.addresses)
    else:
        addresses = DEFAULT_ADDRESSES

    # Parse device IDs if specified
    device_filter_ids = None
    if args.devices:
        device_filter_ids = parse_device_ids(args.devices)
        if not device_filter_ids:
            print("Error: No valid device IDs specified")
            sys.exit(1)

    # Parse core coordinates if specified
    core_filter_coords = None
    if args.cores:
        core_filter_coords = parse_core_coordinates(args.cores)
        if not core_filter_coords:
            print("Error: No valid core coordinates specified")
            sys.exit(1)

    if args.fabric_streams:
        print(f"Fabric stream register mode:")
        print(f"  Stream group: {args.stream_group}")
        group_info = FABRIC_STREAM_GROUPS.get(args.stream_group, {})
        if group_info:
            print(f"  Description: {group_info.get('description', 'N/A')}")
            print(f"  Stream IDs: {group_info.get('stream_ids', [])}")
    elif args.buffer_mode:
        print(f"Reading {len(addresses)} buffer(s):")
        for addr in addresses:
            print(f"  0x{addr:08x}")
        print(f"\n=== BUFFER CONFIGURATION ===")
        print(f"  Elements per buffer: {args.num_elements}")
        print(f"  Slot size: {args.slot_size} bytes")
        print(f"  Data per slot: {args.data_per_slot} bytes")
        print(f"  Total buffer size: {args.num_elements * args.slot_size} bytes")
    else:
        print(f"Reading from {len(addresses)} address(es):")
        for addr in addresses:
            print(f"  0x{addr:08x}")

    # Initialize dumper
    dumper = ERISCDumper()

    # Set filtering options
    filter_active = not args.include_idle  # Default: filter to active only
    check_reset = not args.skip_reset_check  # Default: check reset status

    print(f"\n=== FILTERING OPTIONS ===")
    print(f"  Include idle cores: {args.include_idle}")
    print(f"  Check reset status: {check_reset}")
    if device_filter_ids:
        print(f"  Device filter: {device_filter_ids}")
    if core_filter_coords:
        core_strs = [f"({x},{y})" for x, y in core_filter_coords]
        print(f"  Core filter: {core_strs}")

    if args.fabric_streams:
        # Fabric stream register mode
        print(f"\n=== FABRIC STREAM REGISTER MODE ===")
        results = dumper.read_fabric_stream_registers(
            group_name=args.stream_group,
            filter_active=filter_active,
            check_reset=check_reset,
            device_filter_ids=device_filter_ids,
            core_filter_coords=core_filter_coords,
        )

        if results:
            # Detect architecture from first device for display
            devices = dumper.get_devices()
            arch = "wormhole"  # default
            if devices:
                arch = detect_device_architecture(devices[0])

            dumper.print_fabric_stream_matrix(results, args.stream_group, arch, args.decimal)

        # Handle JSON output for fabric streams
        if args.json:
            output_data = {
                "mode": "fabric_streams",
                "stream_group": args.stream_group,
                "architecture": arch if "arch" in locals() else "unknown",
                "results": results,
            }
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"JSON results written to {args.output}")
            else:
                print("\nJSON Fabric Stream Results:")
                print(json.dumps(output_data, indent=2))

    elif args.poll:
        # Polling mode
        print(f"\n=== POLLING MODE ===")
        results = dumper.poll_values(
            addresses=addresses,
            filter_active=filter_active,
            check_reset=check_reset,
            interval=args.interval,
            duration=args.duration,
            changes_only=args.changes_only,
            csv_output=args.csv,
            output_file=args.output,
            show_delta_summary=not args.no_delta_summary,
            use_decimal=args.decimal,
            device_filter_ids=device_filter_ids,
            core_filter_coords=core_filter_coords,
        )

        # In polling mode, CSV/console output is handled within poll_values
        # Only output JSON summary if requested and not CSV
        if args.json and not args.csv:
            output_data = {
                "mode": "polling",
                "addresses": [f"0x{addr:08x}" for addr in addresses],
                "interval_ms": args.interval * 1000,
                "duration_s": args.duration,
                "sample_count": len(results),
                "results": results,
            }

            if args.output and not args.csv:
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nPolling results written to {args.output}")
            else:
                print("\nJSON Polling Results:")
                print(json.dumps(output_data, indent=2))
    else:
        # Single snapshot mode
        if args.buffer_mode:
            print(f"\n=== BUFFER CONTENTS ===")

        results = dumper.dump_values(
            addresses=addresses,
            filter_active=filter_active,
            check_reset=check_reset,
            buffer_mode=args.buffer_mode,
            num_elements=args.num_elements,
            slot_size=args.slot_size,
            data_per_slot=args.data_per_slot,
            use_decimal=args.decimal,
            device_filter_ids=device_filter_ids,
            core_filter_coords=core_filter_coords,
        )

        # Output results
        if args.json:
            output_data = {"mode": "snapshot", "addresses": [f"0x{addr:08x}" for addr in addresses], "results": results}

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults written to {args.output}")
            else:
                print("\nJSON Results:")
                print(json.dumps(output_data, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
