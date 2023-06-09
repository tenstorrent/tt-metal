#include <array>
#include <vector>

#include "tt_metal/hostdevcommon/common_values.hpp"

using std::array;
using std::vector;

// This is used for relays in which we read a large block of data
// and we want to relay small portions of this data to workers
struct TrailingWriteCommand {
    u32 src;
    u32 dst;
    u32 dst_noc;
    u32 transfer_size;
    u32 num_receivers;
};

// The beginning of data section for dispatcher
static constexpr u32 DEVICE_COMMAND_DATA_ADDR = 150 * 1024;

static constexpr u32 DEVICE_COMMAND_NUM_ENTRIES = 16 + 11 * 104;
static constexpr u32 NUM_ENTRIES_PER_BUFFER_RELAY = 11;
static constexpr u32 CONTROL_SECTION_NUM_ENTRIES = 16;
static constexpr u32 RELAY_BUFFER_NUM_ENTRIES = 4 * NUM_ENTRIES_PER_BUFFER_RELAY;
static constexpr u32
    RELAY_PROGRAM_NUM_ENTRIES =  // Whatever is left of the available size, we allocate for relaying program data
    DEVICE_COMMAND_NUM_ENTRIES - CONTROL_SECTION_NUM_ENTRIES - RELAY_BUFFER_NUM_ENTRIES;

// DeviceCommand.desc organized as follows
// finish (whether we need to notify host when we finished)
// launch (whether we need to notify all worker cores to run)
// num relays (how many buffers are we moving around)
// relay addresses (list with 'num relays' entries specifying
// how to move the buffers around)

// We need to ensure that the command size is divisible by 32
static_assert(DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32) % 32 == 0);

// To stay consistent with the 16B addressing on grayskull, I created this constant
static constexpr u32 NUM_16B_WORDS_IN_DEVICE_COMMAND = (DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32)) / 16;
class DeviceCommand {
   private:
    static constexpr u32 num_4B_words_in_relay_buffer_instruction = 11;
    static constexpr u32 num_possible_relay_buffer_instructions = 4;

    // Command header
    static constexpr u32 finish_idx = 0;
    static constexpr u32 launch_idx = 1;
    static constexpr u32 data_size_in_bytes_idx = 2;
    static constexpr u32 num_relay_buffer_reads_idx = 3;
    static constexpr u32 num_relay_buffer_writes_idx = 4;
    static constexpr u32 num_relay_program_writes_idx = 5;

    // Relay instructions
    static_assert(CONTROL_SECTION_NUM_ENTRIES == 16);
    u32 relay_buffer_entry_idx = CONTROL_SECTION_NUM_ENTRIES;  // Not const, keeps track of which index in the array we're at

    // This magic 16 coming from the fact that we needed to over-allocate the control bit
    // section in order to have the command size be nicely divisble by 32
    static_assert(CONTROL_SECTION_NUM_ENTRIES + RELAY_BUFFER_NUM_ENTRIES == 60);
    u32 relay_program_entry_idx =
        CONTROL_SECTION_NUM_ENTRIES + RELAY_BUFFER_NUM_ENTRIES;

    array<u32, DEVICE_COMMAND_NUM_ENTRIES> desc;

    // Creates a relay instruction in which the first address is a single page and the second can be multiple pages.
    // Num bursts corresponds to how many bursts of data we need to pull into the dispatch core (essentially the number
    // of relays). We try to read in as much data per burst as possible, and if the data is not divisible by num bursts,
    // we have a remainder step in which we try to relay the last chunk, specified by remainder_burst_size.
    void add_buffer_relay(
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
        u32 banking_enum);

   public:
    DeviceCommand();
    static constexpr u32 size() { return DEVICE_COMMAND_NUM_ENTRIES; }
    static constexpr u32 size_in_bytes() { return DEVICE_COMMAND_NUM_ENTRIES * sizeof(u32); }
    // static constexpr u32 relay_buffer_section_offset() { return 6; }
    // static constexpr u32 relay_program_section_offset() { return DeviceCommand::relay_buffer_section_offset() +
    // DeviceCommand::num_possible_relay_buffer_instructions; }

    void finish();  // Creates a finish command, in which the command queue is blocked until the device notifies host of
                    // completion.

    void launch();  // Launches a program

    // 'dst' must be a single bank
    void add_read_buffer_instruction(
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
    void add_write_buffer_instruction(
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

    // The data transfer pattern that this instruction
    // attempts to resolve is when we need to read data
    // such as kernel binaries/cb configs/sem configs/rt args into
    // the dispatch core's L1 in one shot, and then sending
    // small pieces of this data around (multicasting or
    // unicasting) where the transfer sizes are not uniform
    // in size
    void add_read_multi_write_instruction(u32 src, u32 src_noc, u32 transfer_size, vector<TrailingWriteCommand> write_commands);

    // number of bytes in buffer following command, if applicable
    void set_data_size_in_bytes(u32 data_size_in_bytes);

    u32 get_data_size_in_bytes() const;

    const array<u32, DEVICE_COMMAND_NUM_ENTRIES>& get_desc() const;
};
