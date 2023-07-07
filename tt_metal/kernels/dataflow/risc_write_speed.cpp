#include <cstdint>

//#define OPT_WRITE 1

void kernel_main() {
    std::uint32_t   buffer_src_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t   buffer_dst_addr  = get_arg_val<uint32_t>(1);

    std::uint32_t   dst_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t   dst_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t   transaction_size = get_arg_val<uint32_t>(4);
    std::uint32_t   num_transactions = get_arg_val<uint32_t>(5);
    std::uint32_t   num_repetitions  = get_arg_val<uint32_t>(6);

    // Use this reg for cmd buf
    std::uint32_t cmd_buf = NCRISC_WR_REG_CMD_BUF;

    // DRAM NOC src address

    for (std::uint32_t i=0; i<num_repetitions; i++) {

        std::uint32_t buffer_src_addr_  = buffer_src_addr;
        std::uint32_t buffer_dst_addr_  = buffer_dst_addr;
        //constexpr std::uint32_t transaction_size_const = 32; // optionally optimize w/ const transaction size
        std::uint32_t transaction_size_const = transaction_size;

        #ifdef OPT_WRITE
            // reset these to the original value for each repetition
            std::uint64_t buffer_dst_noc_addr = dataflow::get_noc_addr(dst_noc_x, dst_noc_y, buffer_dst_addr);


            noc_fast_write_set_len(transaction_size_const);
            noc_fast_write_set_dst_xy(buffer_dst_noc_addr);
            noc_fast_write_set_cmd_field(NOC_UNICAST_WRITE_VC, false, false);

            for (std::uint32_t j=0; j<num_transactions; j++) {
                noc_fast_write(buffer_src_addr_, buffer_dst_addr_);
                buffer_src_addr_ += transaction_size_const;
                buffer_dst_addr_ += transaction_size_const;
            }
            noc_fast_write_inc_num_dests(num_transactions);

        #else

            for (std::uint32_t j=0; j<num_transactions; j++) {
                std::uint64_t buffer_dst_noc_addr = dataflow::get_noc_addr(dst_noc_x, dst_noc_y, buffer_dst_addr_);
                dataflow::noc_async_write(buffer_src_addr_, buffer_dst_noc_addr, transaction_size_const);
                buffer_src_addr_ += transaction_size_const;
                buffer_dst_addr_ += transaction_size_const;
            }

        #endif

        dataflow::noc_async_write_barrier();
    }

}
