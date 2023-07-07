#include <cstdint>


//#define OPT_READ 1

void kernel_main() {
    std::uint32_t   buffer_dst_addr  = get_arg_val<uint32_t>(0);
    std::uint32_t   buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t   src_noc_x        = get_arg_val<uint32_t>(2);
    std::uint32_t   src_noc_y        = get_arg_val<uint32_t>(3);

    std::uint32_t   transaction_size = get_arg_val<uint32_t>(4);
    std::uint32_t   num_transactions = get_arg_val<uint32_t>(5);
    std::uint32_t   num_repetitions  = get_arg_val<uint32_t>(6);

    // DRAM NOC src address

    for (std::uint32_t i=0; i<num_repetitions; i++) {
        // reset these to the original value for each repetition
        buffer_dst_addr  = get_arg_val<uint32_t>(0);
        buffer_src_addr  = get_arg_val<uint32_t>(1);
        std::uint64_t buffer_src_noc_addr = dataflow::get_noc_addr(src_noc_x, src_noc_y, buffer_src_addr);

        #ifndef OPT_READ

        for (std::uint32_t j=0; j<num_transactions; j++) {


            buffer_src_noc_addr = dataflow::get_noc_addr(src_noc_x, src_noc_y, buffer_src_addr);

            dataflow::noc_async_read(buffer_src_noc_addr, buffer_dst_addr, transaction_size);

            buffer_src_addr += transaction_size;
            buffer_dst_addr += transaction_size;
        }
        // wait all reads from all transactions to be flushed (ie received)
        dataflow::noc_async_read_barrier();

        #else

        std::uint32_t transaction_size_const = transaction_size;

        noc_fast_read_set_len(transaction_size_const);
        buffer_src_noc_addr = dataflow::get_noc_addr(src_noc_x, src_noc_y, buffer_src_addr);
        noc_fast_read_set_src_xy(buffer_src_noc_addr);

        for (std::uint32_t j=0; j<num_transactions; j++) {

            noc_fast_read(buffer_src_addr, buffer_dst_addr);
            buffer_src_addr += transaction_size_const;
            buffer_dst_addr += transaction_size_const;
        }
        noc_fast_read_inc_num_issued(num_transactions);
        // wait all reads from all transactions to be flushed (ie received)
        dataflow::noc_async_read_barrier();

        #endif


    }
}
