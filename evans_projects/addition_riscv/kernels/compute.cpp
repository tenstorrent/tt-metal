void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t src0_l1 = get_arg_val<uint32_t>(3);
    uint32_t src1_l1 = get_arg_val<uint32_t>(4);
    uint32_t dst_l1 = get_arg_val<uint32_t>(5);

    // Create address generators for the input buffers. Consider these the
    // pointers for interleaved buffers. The parameters here must match the
    // parameters used in the host code when creating the buffers.
    // Type is InterleavedAddrGen (PascalCase), not interleaved_addr_gen (namespace).

    InterleavedAddrGen<true> src0 = {.bank_base_address = src0_dram, .page_size = sizeof(uint32_t)};
    InterleavedAddrGen<true> src1 = {.bank_base_address = src1_dram, .page_size = sizeof(uint32_t)};
    InterleavedAddrGen<true> dst = {.bank_base_address = dst_dram, .page_size = sizeof(uint32_t)};

    uint64_t src0_dram_noc_addr = get_noc_addr(0, src0);
    uint64_t src1_dram_noc_addr = get_noc_addr(0, src1);
    uint64_t dst_dram_noc_addr = get_noc_addr(0, dst);

    noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
    noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
    noc_async_read_barrier();

    uint32_t* dat0 = (uint32_t*)src0_l1;
    uint32_t* dat1 = (uint32_t*)src1_l1;
    uint32_t* out = (uint32_t*)dst_l1;

    out[0] = dat0[0] + dat1[0];

    noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));
    noc_async_write_barrier();
}
