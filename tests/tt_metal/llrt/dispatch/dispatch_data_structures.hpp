
struct CopyDescriptor {
    // These two attributes are only used by host to know where
    // to write this copy descriptor
    uint32_t l1_addr;

    /*
        How many reads the dispatch core performs to bring
        things into local L1. Useful for reading in kernel
        hexes, runtime args, cb configs, etc.
    */
    uint32_t num_reads;

    /*
        Vector of tuples containing src (noc addr),
        dst (local l1 addr), and num bytes in transfer.
    */
    const vector<tuple<uint64_t, uint32_t, uint32_t>> reads;

    /*
        How many many writes the dispatch core performs. Useful
        for writing hexes, runtime args, etc. to a remote core.
    */
    uint32_t num_writes;
    vector<tuple<uint32_t, uint64_t, uint32_t>> writes;

    /*
        How many cores to deassert/assert, potentially
        through multicast
    */
    uint32_t num_resets;

    /*
        How many worker cores are running. Can be different
        than num resets since we are potentially multicasting
        the deassert/assert reset values.
    */
   uint32_t num_workers;

    /*
        In here we write the address of the dispatch core.
        To expand on this, think of this as letting
        the worker cores know who sent them data so that
        the worker cores can let the dispatcher know when
        they've finished.
    */
    vector<uint64_t> notifies;

    /*
        The registers in which we write deassert/assert reset masks
        to launch the worker cores.
    */
    vector<uint64_t> resets;
};

ostream &operator<<(ostream &os, const CopyDescriptor &copy_desc) {
    os << "l1_addr: " << copy_desc.l1_addr << "\n"
       << "num_reads: " << copy_desc.num_reads << "\n";

    int i = 0;
    for (tuple<uint64_t, uint32_t, uint32_t> read : copy_desc.reads) {
        os << "read " << i << ": ";
        os << std::get<0>(read) << "(src), ";
        os << std::get<1>(read) << "(dst), ";
        os << std::get<2>(read) << "(size)"
           << "\n";
        i++;
    }

    os << "num_writes: " << copy_desc.num_writes << "\n";

    i = 0;
    for (tuple<uint32_t, uint64_t, uint32_t> write : copy_desc.writes) {
        os << "write " << i << ": ";
        os << std::get<0>(write) << "(src), ";
        os << std::get<1>(write) << "(dst), ";
        os << std::get<2>(write) << "(size)"
           << "\n";
        i++;
    }

    os << "num_resets: " << copy_desc.num_resets << "\n";
    os << "num_workers: " << copy_desc.num_workers << "\n";
    i = 0;
    for (uint64_t reset_addr : copy_desc.resets) {
        os << "reset " << i << ": " << reset_addr << "\n";
        i++;
    }
    return os;
}

struct DramConfig {
    const vector<vector<uint32_t>> kernels;
    const vector<vector<uint32_t>> runtime_args;
    const vector<vector<uint32_t>> cb_configs;

    const vector<pair<uint32_t, uint32_t>> kernel_addrs_and_sizes;
    const vector<pair<uint32_t, uint32_t>> runtime_args_addrs_and_sizes;
    const vector<pair<uint32_t, uint32_t>> cb_configs_addrs_and_sizes;
};

ostream &operator<<(ostream &os, const DramConfig &dram_config) {
    os << "kernel_addrs_and_sizes: [";
    for (const auto &[kernel_addr, kernel_size] : dram_config.kernel_addrs_and_sizes) {
        os << "(" << kernel_addr << ", " << kernel_size << "), ";
    }
    os << "\b\b]\n";

    os << "runtime_args_addrs_and_sizes: [";
    for (const auto &[runtime_args_addr, runtime_args_size] : dram_config.runtime_args_addrs_and_sizes) {
        os << "(" << runtime_args_addr << ", " << runtime_args_size << "), ";
    }
    os << "\b\b]\n";

    os << "cb_configs_addrs_and_sizes: [";
    for (const auto &[cb_config_addr, cb_config_size] : dram_config.cb_configs_addrs_and_sizes) {
        os << "(" << cb_config_addr << ", " << cb_config_size << "), ";
    }
    os << "\b\b]\n";
    return os;
}
