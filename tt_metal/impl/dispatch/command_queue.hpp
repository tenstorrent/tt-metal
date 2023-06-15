
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/dispatch/thread_safe_queue.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/program.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_parameters.h"

using namespace tt::tt_metal;
using std::pair;
using std::set;
using std::shared_ptr;
using std::tuple;
using std::unique_ptr;

enum class TransferType : u8 {
    // RISCV types
    B = 0,
    N = 1,
    T0 = 2,
    T1 = 3,
    T2 = 4,

    // CB and Sems
    CB = 5,
    SEM = 6,
};

typedef tuple<
    u32 /* addr */,
    u32 /* start_in_bytes */,
    u32 /* kernel_size_in_bytes */,
    u32 /* noc_multicast_encoding */,
    u32 /* num_receivers */>
    transfer_info;
struct ProgramSection {
    // Maps type to src, transfer size, and multicast encoding
    map<TransferType, vector<transfer_info>> section;  // Maps the RISC-V type to transfer info
    size_t size_in_bytes;

    vector<transfer_info>& at(TransferType key) { return this->section.at(key); }
};

// The role of this datastructure is to essentially describe
// the mapping between binaries within DRAM to worker cores.
// Given that our program buffer could potentially be bigger
// than available L1, we may need
struct ProgramSrcToDstAddrMap {
    vector<u32> program_vector;
    vector<ProgramSection> program_sections;
    vector<u32> worker_noc_coords;
};

ProgramSrcToDstAddrMap ConstructProgramSrcToDstAddrMap(const Device* device, Program& program);

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER, ENQUEUE_PROGRAM, FINISH, INVALID };

string EnqueueCommandTypeToString(EnqueueCommandType ctype);

// TEMPORARY! TODO(agrebenisan): need to use proper macro based on loading noc
#define NOC_X(x) x
#define NOC_Y(y) y

#define NOC_MULTICAST_ENCODING(x_start, y_start, x_end, y_end)                                          \
    ((x_start) << (2 * NOC_ADDR_NODE_ID_BITS)) | ((y_start) << (3 * NOC_ADDR_NODE_ID_BITS)) | (x_end) | \
        ((y_end) << (NOC_ADDR_NODE_ID_BITS))

u32 noc_coord_to_u32(CoreCoord coord);

class Command {
    EnqueueCommandType type_ = EnqueueCommandType::INVALID;

   public:
    Command() {}
    virtual void process(){};
    virtual EnqueueCommandType type() = 0;
    virtual const DeviceCommand assemble_device_command(u32 buffer_size) = 0;
};

class EnqueueReadBufferCommand : public Command {
   private:
    Device* device;
    SystemMemoryWriter& writer;
    vector<u32>& dst;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_READ_BUFFER;

   public:
    Buffer& buffer;
    u32 read_buffer_addr;
    EnqueueReadBufferCommand(Device* device, Buffer& buffer, vector<u32>& dst, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32 dst);

    void process();

    EnqueueCommandType type();
};

class EnqueueWriteBufferCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;

    SystemMemoryWriter& writer;
    vector<u32>& src;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_WRITE_BUFFER;

   public:
    EnqueueWriteBufferCommand(Device* device, Buffer& buffer, vector<u32>& src, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32 src_address);

    void process();

    EnqueueCommandType type();
};

class EnqueueProgramCommand : public Command {
   private:
    Device* device;
    Buffer& buffer;
    ProgramSrcToDstAddrMap& program_to_dev_map;
    RuntimeArgs rt_args;
    SystemMemoryWriter& writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::ENQUEUE_PROGRAM;

   public:
    static map<const Program*, DeviceCommand> command_cache;
    EnqueueProgramCommand(Device*, Buffer&, ProgramSrcToDstAddrMap&, SystemMemoryWriter&, RuntimeArgs);

    const DeviceCommand assemble_device_command(u32);

    void process();

    EnqueueCommandType type();
};

// Easiest way for us to process finish is to explicitly have the device
// write to address chosen by us for finish... that way we don't need
// to mess with checking recv and acked
class FinishCommand : public Command {
   private:
    Device* device;
    SystemMemoryWriter& writer;
    static constexpr EnqueueCommandType type_ = EnqueueCommandType::FINISH;

   public:
    FinishCommand(Device* device, SystemMemoryWriter& writer);

    const DeviceCommand assemble_device_command(u32);

    void process();

    EnqueueCommandType type();
};

void send_dispatch_kernel_to_device(Device* device);

class CommandQueue {
   public:
    CommandQueue(Device* device);

    ~CommandQueue();

   private:
    Device* device;
    SystemMemoryWriter sysmem_writer;
    TSQueue<shared_ptr<Command>>
        processing_thread_queue;  // These are commands that have not been placed in system memory
    // thread processing_thread;
    map<const Program*, unique_ptr<Buffer>>
        program_to_buffer;  // Using raw pointer since I want to be able to hash program inexpensively. This implies
                            // program object cannot be destroyed during the lifetime of the user's program

    map<const Program*, ProgramSrcToDstAddrMap> program_to_dev_map;

    void enqueue_command(shared_ptr<Command> command, bool blocking);

    void enqueue_read_buffer(Buffer& buffer, vector<u32>& dst, bool blocking);

    void enqueue_write_buffer(Buffer& buffer, vector<u32>& src, bool blocking);

    void enqueue_program(Program& program, const RuntimeArgs& runtime_args, bool blocking);

    void finish();

    friend void EnqueueReadBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& dst, bool blocking);
    friend void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buffer, vector<u32>& src, bool blocking);
    friend void EnqueueProgram(CommandQueue& cq, Program& program, const RuntimeArgs& runtime_args, bool blocking);
    friend void Finish(CommandQueue& cq);
};
