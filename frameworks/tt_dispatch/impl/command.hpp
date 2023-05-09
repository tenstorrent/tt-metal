#include <array>
#include <vector>

#include "tt_metal/hostdevcommon/common_values.hpp"

using std::array;
using std::vector;

// DeviceCommand.desc orgranizes as follows
// finish (whether we need to notify host when we finished)
// launch (whether we need to notify all worker cores to run)
// num relays (how many buffers are we moving around)
// relay addresses (list with 'num relays' entries specifying
// how to move the buffers around)
static constexpr u32 DeviceCommandNumEntries = 6 + 11 * 110;
static constexpr u32 NUM_16B_WORDS_IN_COMMAND_TABLE = (DeviceCommandNumEntries * 4) / 16;
class DeviceCommand {
   private:
    const u32 finish_idx = 0;
    const u32 launch_idx = 1;
    const u32 data_size_in_bytes_idx = 2;
    const u32 num_reads_idx = 3;
    const u32 num_writes_idx = 4;
    u32 relay_entry_idx = 5;                   // Not const, keeps track of which index in the array we're at

    array<u32, DeviceCommandNumEntries> desc;

    // Creates a relay instruction in which the first address is a single page and the second can be multiple pages.
    // Num bursts corresponds to how many bursts of data we need to pull into the dispatch core (essentially the number
    // of relays). We try to read in as much data per burst as possible, and if the data is not divisible by num bursts,
    // we have a remainder step in which we try to relay the last chunk, specified by remainder_burst_size.
    void add_relay(
        u32 addr0,
        u32 addr0_noc,
        u32 addr1,
        u32 addr1_noc_start,
        u32 num_bursts,
        u32 burst_size,
        u32 num_pages_per_burst,
        u32 page_size,
        u32 remainder_burst_size,
        u32 num_pages_per_remainder_burst,
        u32 banking_enum) {
        this->desc[this->relay_entry_idx] = addr0;
        this->desc[this->relay_entry_idx + 1] = addr0_noc;
        this->desc[this->relay_entry_idx + 2] = addr1;
        this->desc[this->relay_entry_idx + 3] = addr1_noc_start;
        this->desc[this->relay_entry_idx + 4] = num_bursts;
        this->desc[this->relay_entry_idx + 5] = burst_size;
        this->desc[this->relay_entry_idx + 6] = num_pages_per_burst;
        this->desc[this->relay_entry_idx + 7] = page_size;
        this->desc[this->relay_entry_idx + 8] = remainder_burst_size;
        this->desc[this->relay_entry_idx + 9] = num_pages_per_remainder_burst;
        this->desc[this->relay_entry_idx + 10] = banking_enum;
        this->relay_entry_idx += 11;
    }

   public:
    DeviceCommand();
    static constexpr u32 size() { return DeviceCommandNumEntries; }
    static constexpr u32 size_in_bytes() { return DeviceCommandNumEntries * sizeof(u32); }

    void finish();  // Creates a finish command, in which the command queue is blocked until the device notifies host of
                    // completion.

    void launch();  // Launches a program

    // 'dst' must be a single bank
    void add_read_relay(
        u32 dst,
        u32 dst_noc,
        u32 src,
        u32 src_noc_start,
        u32 num_bursts,
        u32 burst_size,
        u32 num_pages_per_burst,
        u32 page_size,
        u32 remainder_burst_size,
        u32 num_pages_per_remainder_burst,
        u32 banking_enum);

    // 'src' must be a single bank
    void add_write_relay(
        u32 src,
        u32 src_noc,
        u32 dst,
        u32 dst_noc_start,
        u32 num_bursts,
        u32 burst_size,
        u32 num_pages_per_burst,
        u32 page_size,
        u32 remainder_burst_size,
        u32 num_pages_per_remainder_burst,
        u32 banking_enum);
    // number of bytes in buffer following command, if applicable
    void set_data_size_in_bytes(u32 data_size_in_bytes);

    const array<u32, DeviceCommandNumEntries>& get_desc() const;
};
