#!/usr/bin/env python3
"""
Simulation of C++ DRAM bank addressing for sharded tensors.

This script simulates the address calculation performed by Buffer::page_address()
in tt_metal/impl/buffers/buffer.cpp (lines 549-594).

Reference C++ code:
    DeviceAddr Buffer::page_address(DeviceAddr bank_id, DeviceAddr page_index) const {
        DeviceAddr num_banks = allocator_->get_num_banks(buffer_type_);
        DeviceAddr pages_offset_within_bank = page_index / num_banks;
        auto offset = (round_up(page_size(), alignment()) * pages_offset_within_bank);
        return translate_page_address(offset, bank_id);
    }

    DeviceAddr Buffer::translate_page_address(DeviceAddr offset, uint32_t bank_id) const {
        DeviceAddr base_page_address = this->address() + allocator_->get_bank_offset(buffer_type_, bank_id);
        return base_page_address + offset;
    }

Formula:
    page_address(bank_id, page_index) =
        base_address + bank_offset[bank_id] + (aligned_page_size * (page_index / num_banks))
"""


def round_up(value, multiple):
    """Round up to nearest multiple (C++ round_up function)."""
    return ((value + multiple - 1) // multiple) * multiple


def simulate_page_address(base_address, bank_offsets, page_size, alignment, num_banks, bank_id, page_index):
    """
    Simulate C++ Buffer::page_address() calculation.

    Args:
        base_address: Base buffer address (from buffer->address())
        bank_offsets: List of bank offset values [offset_0, offset_1, ..., offset_7]
        page_size: Size of one page in bytes
        alignment: Memory alignment requirement
        num_banks: Total number of DRAM banks
        bank_id: Which bank (0-7 for Blackhole)
        page_index: Logical page index (round-robin across all banks)

    Returns:
        Physical NOC address for the page in that bank
    """
    # Calculate offset within this bank
    pages_offset_within_bank = page_index // num_banks  # Integer division
    aligned_page_size = round_up(page_size, alignment)
    offset = aligned_page_size * pages_offset_within_bank

    # Translate to physical address
    bank_offset = bank_offsets[bank_id]
    physical_address = base_address + bank_offset + offset

    return physical_address, aligned_page_size, pages_offset_within_bank, offset


def main():
    print("=" * 80)
    print("DRAM Bank Address Simulation")
    print("=" * 80)

    # ============================================================================
    # Configuration (matching test_banks in test_disaggregation.py)
    # ============================================================================

    # Tensor: [1, 1, 256, 576]
    # Sharding: 8 chunks of [1, 1, 32, 576] each
    # Distribution: ROUND_ROBIN_1D across 8 DRAM banks

    num_banks = 8
    tokens_per_chunk = 32
    head_dim = 576
    total_tokens = 256

    # From test output: base address = 0x493E40 = 4800064
    base_address = 0x493E40

    # Page size calculation:
    # Each chunk: [1, 1, 32, 576] in bfloat8_b (1 byte per element)
    # Tiled layout: round up to tile boundaries (32x32 tiles)
    # Width: 576 elements → 18 tiles of 32 elements each
    # Height: 32 elements → 1 tile of 32 elements
    # Total: 18 tiles
    # Tile size in bfp8: 1088 bytes (32×32 elements + 4 bytes exponent per 32 elements)
    # Page size ≈ 18 × 1088 = 19584 bytes

    page_size = 19584
    alignment = 32  # DRAM alignment (typical)

    # HYPOTHETICAL bank offsets for demonstration
    # (These are device-specific constants from SOC descriptor)
    # In reality, these come from allocator_->get_bank_offset(BufferType::DRAM, bank_id)
    # For Blackhole, these might be offsets to different DRAM channels

    # Hypothetical example: banks at 0x10000000 intervals
    hypothetical_bank_offsets = [
        0x00000000,  # Bank 0
        0x10000000,  # Bank 1
        0x20000000,  # Bank 2
        0x30000000,  # Bank 3
        0x40000000,  # Bank 4
        0x50000000,  # Bank 5
        0x60000000,  # Bank 6
        0x70000000,  # Bank 7
    ]

    print(f"\nConfiguration:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Tokens per chunk: {tokens_per_chunk}")
    print(f"  Number of banks: {num_banks}")
    print(f"  Number of chunks: {total_tokens // tokens_per_chunk}")
    print(f"  Distribution: ROUND_ROBIN_1D")
    print(f"  Base address: 0x{base_address:016X} ({base_address})")
    print(f"  Page size: {page_size} bytes")
    print(f"  Alignment: {alignment} bytes")

    print(f"\nBank offsets (HYPOTHETICAL for demonstration):")
    for bank_id, offset in enumerate(hypothetical_bank_offsets):
        print(f"  Bank {bank_id}: 0x{offset:08X}")

    # ============================================================================
    # Round-robin distribution
    # ============================================================================

    print(f"\nRound-robin page distribution:")
    print(f"  Page 0 → Bank 0 (tokens 0-31)")
    print(f"  Page 1 → Bank 1 (tokens 32-63)")
    print(f"  Page 2 → Bank 2 (tokens 64-95)")
    print(f"  ...")
    print(f"  Page 7 → Bank 7 (tokens 224-255)")

    # ============================================================================
    # Address calculation for each bank
    # ============================================================================

    print(f"\n" + "=" * 80)
    print(f"Address Calculation (using hypothetical bank offsets)")
    print(f"=" * 80)
    print(f"\n{'Bank':<6} {'Page':<6} {'Offset':<12} {'Address':<20} {'Token Range'}")
    print(f"{'-'*6} {'-'*6} {'-'*12} {'-'*20} {'-'*15}")

    for bank_id in range(num_banks):
        # In round-robin with 8 pages and 8 banks: page_index = bank_id
        page_index = bank_id

        address, aligned_page_size, pages_within_bank, offset_value = simulate_page_address(
            base_address=base_address,
            bank_offsets=hypothetical_bank_offsets,
            page_size=page_size,
            alignment=alignment,
            num_banks=num_banks,
            bank_id=bank_id,
            page_index=page_index,
        )

        token_start = bank_id * tokens_per_chunk
        token_end = token_start + tokens_per_chunk - 1

        print(
            f"{bank_id:<6} {page_index:<6} {offset_value:<12} 0x{address:016X}   " f"[{token_start:3d}-{token_end:3d}]"
        )

    # ============================================================================
    # Special case analysis for this test
    # ============================================================================

    print(f"\n" + "=" * 80)
    print(f"Special Case: 8 Pages, 8 Banks (This Test)")
    print(f"=" * 80)

    print(f"\nFor each bank_id from 0 to 7:")
    print(f"  page_index = bank_id  (round-robin)")
    print(f"  pages_offset_within_bank = page_index / num_banks")
    print(f"                          = bank_id / 8")
    print(f"                          = 0  (integer division)")
    print(f"  offset = aligned_page_size * 0 = 0")
    print(f"")
    print(f"  Therefore:")
    print(f"  chunk_address[bank_id] = base_address + bank_offset[bank_id] + 0")
    print(f"                         = base_address + bank_offset[bank_id]")

    print(f"\nActual addresses (with REAL bank offsets):")
    print(f"  Bank 0: 0x{base_address:016X} + bank_offset[0]")
    print(f"  Bank 1: 0x{base_address:016X} + bank_offset[1]")
    print(f"  ...")
    print(f"  Bank 7: 0x{base_address:016X} + bank_offset[7]")

    print(f"\n" + "=" * 80)
    print(f"Key Insights")
    print(f"=" * 80)
    print(f"1. Base address (0x{base_address:016X}) is the buffer's starting address")
    print(f"2. Each bank has a hardware-defined offset (from SOC descriptor)")
    print(f"3. In this case, all chunks start at their bank's base (no intra-bank offset)")
    print(f"4. Bank offsets are NOT accessible from Python currently")
    print(f"5. To get actual addresses, you need C++ Buffer::page_address() API")
    print(f"\nC++ APIs to get bank offsets:")
    print(f"  - Buffer::page_address(bank_id, page_index)")
    print(f"  - Allocator::get_bank_offset(BufferType::DRAM, bank_id)")
    print(f"=" * 80)


if __name__ == "__main__":
    main()
