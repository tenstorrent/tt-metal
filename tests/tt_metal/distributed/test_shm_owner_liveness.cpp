// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A descriptor file or counter segment left behind by a dead owner must look
// "not yet published" to a connector, so the connector keeps waiting for the
// owner's successor instead of attaching to stale state. No device needed.

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include <internal/service/inter_process_counter_channel.hpp>
#include <tt-metalium/experimental/sockets/d2h_stream_service_descriptor.hpp>
#include <tt-metalium/experimental/sockets/h2d_stream_service_descriptor.hpp>
#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>
#include "tt_metal/distributed/inter_process_counter_layout.hpp"
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>
#include "tt_metal/distributed/shm_owner_liveness.hpp"

namespace tt::tt_metal::distributed {
namespace {

using ::testing::HasSubstr;

// A pid that was alive a moment ago and is now gone: fork a child that exits at once, then reap it.
pid_t dead_pid() {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return pid;
}

std::string unique_descriptor_path(const char* stem) {
    static std::atomic<uint32_t> counter{0};
    return fmt::format("/dev/shm/tt_test_{}_{}_{}.bin", stem, getpid(), counter.fetch_add(1));
}

std::string unique_segment_name(const char* stem) {
    static std::atomic<uint32_t> counter{0};
    return fmt::format("/tt_test_{}_{}_{}", stem, getpid(), counter.fetch_add(1));
}

struct ScopedFile {
    std::string path;
    ~ScopedFile() { std::remove(path.c_str()); }
};

struct ScopedSegment {
    std::string name;
    ~ScopedSegment() { ::shm_unlink(name.c_str()); }
};

// A socket descriptor as an owner with the given identity would export it (shm_name embeds the pid).
HDSocketDescriptor socket_descriptor_owned_by(pid_t owner_pid, uint64_t owner_start_time) {
    HDSocketDescriptor desc;
    desc.socket_type = "h2d";
    desc.shm_name = fmt::format("/tt_h2d_{}_7_0", owner_pid);
    desc.shm_size = 4096;
    desc.fifo_size = 1024;
    desc.mesh_coord = {0, 0};
    desc.owner_start_time = owner_start_time;
    return desc;
}

H2DStreamServiceDescriptor h2d_service_descriptor_owned_by(pid_t owner_pid, uint64_t owner_start_time) {
    H2DStreamServiceDescriptor desc;
    desc.global_shape = tt::tt_metal::Shape({1, 64});
    desc.global_dtype = DataType::BFLOAT16;
    desc.mesh_shape = MeshShape(1, 1);
    desc.mapper_config.placements.push_back(MeshMapperConfig::Replicate{});
    desc.mapper_config.placements.push_back(MeshMapperConfig::Replicate{});
    desc.socket_page_size = 128;
    desc.num_socket_pages = 1;
    desc.per_coord_entries.emplace_back(MeshCoordinate(0, 0), socket_descriptor_owned_by(owner_pid, owner_start_time));
    return desc;
}

D2HStreamServiceDescriptor d2h_service_descriptor_owned_by(pid_t owner_pid, uint64_t owner_start_time) {
    D2HStreamServiceDescriptor desc;
    desc.global_shape = tt::tt_metal::Shape({1, 64});
    desc.global_dtype = DataType::BFLOAT16;
    desc.mesh_shape = MeshShape(1, 1);
    desc.mapper_config.placements.push_back(MeshMapperConfig::Replicate{});
    desc.mapper_config.placements.push_back(MeshMapperConfig::Replicate{});
    desc.composer_config.dims.push_back(0);
    desc.composer_config.dims.push_back(1);
    desc.socket_page_size = 128;
    desc.num_socket_pages = 1;
    auto socket = socket_descriptor_owned_by(owner_pid, owner_start_time);
    socket.socket_type = "d2h";
    desc.per_coord_entries.emplace_back(MeshCoordinate(0, 0), std::move(socket));
    return desc;
}

// A counter segment exactly as an owner with the given identity would leave it (owner_pid 0 = unstamped).
void create_raw_segment(const std::string& name, uint32_t owner_pid, uint64_t owner_start_time) {
    const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    ASSERT_NE(fd, -1) << "shm_open failed for " << name;
    ASSERT_EQ(::ftruncate(fd, sizeof(InterProcessCounterSegment)), 0);
    void* mapped = ::mmap(nullptr, sizeof(InterProcessCounterSegment), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(mapped, MAP_FAILED);
    auto* seg = static_cast<InterProcessCounterSegment*>(mapped);
    seg->prior_clean_shutdown = 1;
    seg->owner_pid = owner_pid;
    seg->owner_start_time = owner_start_time;
    ::munmap(seg, sizeof(InterProcessCounterSegment));
    ::close(fd);
}

template <typename Fn>
void expect_throws_containing(const Fn& fn, const char* needle) {
    try {
        fn();
        FAIL() << "expected an exception containing: " << needle;
    } catch (const std::exception& e) {
        EXPECT_THAT(e.what(), HasSubstr(needle));
    }
}

// ---------------------------------------------------------------------------
// Owner-identity helpers
// ---------------------------------------------------------------------------

TEST(ShmOwnerLiveness, PidFromShmNameParsesNamedShmNames) {
    EXPECT_EQ(pid_from_shm_name("/tt_h2d_4242_987_3"), 4242);
    EXPECT_EQ(pid_from_shm_name("tt_d2h_17_5_0"), 17);
    EXPECT_EQ(pid_from_shm_name("/tt_layer_ack_12"), 0);
    EXPECT_EQ(pid_from_shm_name("/tt_socket_manifest_12"), 0);
    EXPECT_EQ(pid_from_shm_name("/other"), 0);
    EXPECT_EQ(pid_from_shm_name(""), 0);
}

TEST(ShmOwnerLiveness, ProcessStartTimeNamesOneProcessInstance) {
    const uint64_t mine = process_start_time(getpid());
    EXPECT_NE(mine, 0u);
    EXPECT_EQ(mine, process_start_time(getpid()));
    EXPECT_EQ(process_start_time(0), 0u);
    EXPECT_EQ(process_start_time(dead_pid()), 0u);
}

TEST(ShmOwnerLiveness, ProcessStartTimeSurvivesSpacesAndParenthesesInComm) {
    const uint64_t before = process_start_time(getpid());
    char original[17] = {};
    ASSERT_EQ(::prctl(PR_GET_NAME, original), 0);
    ASSERT_EQ(::prctl(PR_SET_NAME, "a) b (c"), 0);
    const uint64_t renamed = process_start_time(getpid());
    ::prctl(PR_SET_NAME, original);

    EXPECT_NE(before, 0u);
    EXPECT_EQ(renamed, before);
}

TEST(ShmOwnerLiveness, IsProcessAliveRejectsDeadAndReusedPids) {
    const pid_t self = getpid();
    const uint64_t mine = process_start_time(self);

    EXPECT_TRUE(is_process_alive(self, 0));
    EXPECT_TRUE(is_process_alive(self, mine));
    // Same pid, different start time: a reused pid, not the original owner.
    EXPECT_FALSE(is_process_alive(self, mine + 1));
    EXPECT_FALSE(is_process_alive(dead_pid(), 0));
}

TEST(ShmOwnerLiveness, OpenFailuresOtherThanMissingAreReported) {
    // A path below a regular file cannot be opened (ENOTDIR). That is not "not published yet" and
    // must surface immediately instead of after the publication timeout.
    const std::string file = unique_descriptor_path("notdir");
    {
        std::ofstream ofs(file);
        ofs << "x";
    }
    const std::string below = file + "/descriptor.bin";
    expect_throws_containing([&] { HDSocketDescriptor::wait_and_read(below, "h2d", 1'000); }, "Not a directory");
    expect_throws_containing([&] { H2DStreamServiceDescriptor::wait_and_read(below, 1'000); }, "Not a directory");
    expect_throws_containing([&] { D2HStreamServiceDescriptor::wait_and_read(below, 1'000); }, "Not a directory");
    std::remove(file.c_str());
}

TEST(ShmOwnerLiveness, ZombieOwnerCountsAsDead) {
    // An owner that exited but was not reaped yet still answers kill(2) and still
    // has a /proc entry with its original start time; it is a dead owner all the same.
    signal(SIGCHLD, SIG_DFL);  // SIG_IGN would make the kernel reap the child for us
    const pid_t child = fork();
    ASSERT_GE(child, 0) << std::strerror(errno);
    if (child == 0) {
        _exit(0);
    }
    const uint64_t start = process_start_time(child);
    bool dead = false;
    for (int i = 0; i < 5000 && !dead; ++i) {  // the child needs a moment to exit
        dead = !is_process_alive(child, start);
        if (!dead) {
            usleep(1000);
        }
    }
    EXPECT_TRUE(dead) << "exited, unreaped child still counted as alive";
    EXPECT_EQ(kill(child, 0), 0) << "the child was reaped by someone else, so the zombie case was not exercised";
    EXPECT_FALSE(ShmResourceTracker::is_pid_alive(child));

    int status = 0;
    waitpid(child, &status, 0);
    EXPECT_FALSE(is_process_alive(child, start));
}

// State letter of the thread-group leader as /proc/<pid>/stat reports it, '\0' if unreadable.
char proc_state_letter(pid_t pid) {
    std::ifstream ifs(fmt::format("/proc/{}/stat", pid));
    std::string line;
    if (!std::getline(ifs, line)) {
        return '\0';
    }
    const auto comm_end = line.rfind(')');
    return (comm_end == std::string::npos || comm_end + 2 >= line.size()) ? '\0' : line[comm_end + 2];
}

void* pause_forever(void*) {
    for (;;) {
        pause();
    }
    return nullptr;
}

TEST(ShmOwnerLiveness, LeaderZombieWithLiveThreadsIsAlive) {
    // A process whose main thread exited while another thread keeps running shows 'Z' in
    // /proc/<pid>/stat for the rest of its life. It is alive. The leader leaves through the
    // raw exit syscall (only that thread ends; pthread_exit would unwind through gtest's frames).
    signal(SIGCHLD, SIG_DFL);
    const pid_t child = fork();
    ASSERT_GE(child, 0) << std::strerror(errno);
    if (child == 0) {
        pthread_t worker;
        if (pthread_create(&worker, nullptr, pause_forever, nullptr) != 0) {
            _exit(2);
        }
        syscall(SYS_exit, 0);
        _exit(3);
    }
    const uint64_t start = process_start_time(child);
    bool leader_zombie = false;
    for (int i = 0; i < 5000 && !leader_zombie; ++i) {
        leader_zombie = proc_state_letter(child) == 'Z';
        if (!leader_zombie) {
            usleep(1000);
        }
    }
    EXPECT_TRUE(leader_zombie) << "could not reach the leader-zombie state";
    EXPECT_TRUE(ShmResourceTracker::is_pid_alive(child));
    EXPECT_TRUE(is_process_alive(child, start));

    kill(child, SIGKILL);
    int status = 0;
    waitpid(child, &status, 0);
    EXPECT_FALSE(is_process_alive(child, start));
}

// ---------------------------------------------------------------------------
// HDSocketDescriptor
// ---------------------------------------------------------------------------

TEST(ShmOwnerLiveness, SocketDescriptorRoundTripsOwnerStartTime) {
    ScopedFile file{unique_descriptor_path("socket")};
    const uint64_t mine = process_start_time(getpid());
    socket_descriptor_owned_by(getpid(), mine).write_to_file(file.path);

    const auto desc = HDSocketDescriptor::read_from_file(file.path);
    EXPECT_EQ(desc.shm_name, fmt::format("/tt_h2d_{}_7_0", getpid()));
    EXPECT_EQ(desc.owner_start_time, mine);
    EXPECT_TRUE(desc.owner_alive());
}

TEST(ShmOwnerLiveness, SocketDescriptorFromLiveOwnerIsRead) {
    ScopedFile file{unique_descriptor_path("socket")};
    socket_descriptor_owned_by(getpid(), process_start_time(getpid())).write_to_file(file.path);

    const auto desc = HDSocketDescriptor::wait_and_read(file.path, "h2d", 1000);
    EXPECT_EQ(desc.fifo_size, 1024u);
}

TEST(ShmOwnerLiveness, SocketDescriptorWithoutStartTimeFallsBackToPidCheck) {
    ScopedFile file{unique_descriptor_path("socket")};
    socket_descriptor_owned_by(getpid(), /*owner_start_time=*/0).write_to_file(file.path);

    EXPECT_NO_THROW(HDSocketDescriptor::wait_and_read(file.path, "h2d", 1000));
}

TEST(ShmOwnerLiveness, SocketDescriptorFromDeadOwnerIsNotPublished) {
    ScopedFile file{unique_descriptor_path("socket")};
    socket_descriptor_owned_by(dead_pid(), /*owner_start_time=*/0).write_to_file(file.path);

    expect_throws_containing(
        [&] { HDSocketDescriptor::wait_and_read(file.path, "h2d", 50); },
        "Timeout waiting for descriptor file to be created");
}

TEST(ShmOwnerLiveness, SocketDescriptorFromReusedPidIsNotPublished) {
    ScopedFile file{unique_descriptor_path("socket")};
    // Our own pid, but a start time that is not ours: the owner died and its pid came back.
    socket_descriptor_owned_by(getpid(), process_start_time(getpid()) + 1).write_to_file(file.path);

    expect_throws_containing(
        [&] { HDSocketDescriptor::wait_and_read(file.path, "h2d", 50); },
        "Timeout waiting for descriptor file to be created");
}

TEST(ShmOwnerLiveness, SocketDescriptorRepublishedByLiveOwnerIsPickedUp) {
    ScopedFile file{unique_descriptor_path("socket")};
    socket_descriptor_owned_by(dead_pid(), /*owner_start_time=*/0).write_to_file(file.path);

    std::thread successor([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        socket_descriptor_owned_by(getpid(), process_start_time(getpid())).write_to_file(file.path);
    });
    const auto desc = HDSocketDescriptor::wait_and_read(file.path, "h2d", 5000);
    successor.join();

    EXPECT_EQ(desc.shm_name, fmt::format("/tt_h2d_{}_7_0", getpid()));
    EXPECT_TRUE(desc.owner_alive());
}

TEST(ShmOwnerLiveness, SocketDescriptorRemovedWhileWaitingIsNotAnError) {
    ScopedFile file{unique_descriptor_path("socket")};
    socket_descriptor_owned_by(dead_pid(), /*owner_start_time=*/0).write_to_file(file.path);

    // What the owner's successor does: its stale scan removes the dead owner's file, then it publishes its own.
    std::thread successor([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::remove(file.path.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        socket_descriptor_owned_by(getpid(), process_start_time(getpid())).write_to_file(file.path);
    });
    const auto desc = HDSocketDescriptor::wait_and_read(file.path, "h2d", 5000);
    successor.join();

    EXPECT_TRUE(desc.owner_alive());
}

// ---------------------------------------------------------------------------
// Stream service descriptors (owner identity travels in the embedded socket descriptors)
// ---------------------------------------------------------------------------

TEST(ShmOwnerLiveness, H2DServiceDescriptorHonoursOwnerLiveness) {
    ScopedFile file{unique_descriptor_path("h2d_service")};
    h2d_service_descriptor_owned_by(dead_pid(), /*owner_start_time=*/0).write_to_file(file.path);
    expect_throws_containing(
        [&] { H2DStreamServiceDescriptor::wait_and_read(file.path, 50); },
        "Timeout waiting for service descriptor file");

    const uint64_t mine = process_start_time(getpid());
    h2d_service_descriptor_owned_by(getpid(), mine).write_to_file(file.path);
    const auto desc = H2DStreamServiceDescriptor::wait_and_read(file.path, 1000);
    ASSERT_EQ(desc.per_coord_entries.size(), 1u);
    EXPECT_EQ(desc.per_coord_entries.front().second.owner_start_time, mine);
}

TEST(ShmOwnerLiveness, D2HServiceDescriptorHonoursOwnerLiveness) {
    ScopedFile file{unique_descriptor_path("d2h_service")};
    d2h_service_descriptor_owned_by(dead_pid(), /*owner_start_time=*/0).write_to_file(file.path);
    expect_throws_containing(
        [&] { D2HStreamServiceDescriptor::wait_and_read(file.path, 50); },
        "Timeout waiting for D2H service descriptor file");

    const uint64_t mine = process_start_time(getpid());
    d2h_service_descriptor_owned_by(getpid(), mine).write_to_file(file.path);
    const auto desc = D2HStreamServiceDescriptor::wait_and_read(file.path, 1000);
    ASSERT_EQ(desc.per_coord_entries.size(), 1u);
    EXPECT_EQ(desc.per_coord_entries.front().second.owner_start_time, mine);
}

// ---------------------------------------------------------------------------
// InterProcessCounterChannel
// ---------------------------------------------------------------------------

TEST(ShmOwnerLiveness, CounterChannelOwnerStampsItsIdentity) {
    ScopedSegment segment{unique_segment_name("ack")};
    InterProcessCounterChannel owner(segment.name);

    const int fd = ::shm_open(segment.name.c_str(), O_RDONLY, 0);
    ASSERT_NE(fd, -1);
    void* mapped = ::mmap(nullptr, sizeof(InterProcessCounterSegment), PROT_READ, MAP_SHARED, fd, 0);
    ASSERT_NE(mapped, MAP_FAILED);
    const auto* seg = static_cast<const InterProcessCounterSegment*>(mapped);
    EXPECT_EQ(seg->owner_pid, static_cast<uint32_t>(getpid()));
    EXPECT_EQ(seg->owner_start_time, process_start_time(getpid()));
    ::munmap(mapped, sizeof(InterProcessCounterSegment));
    ::close(fd);
}

TEST(ShmOwnerLiveness, CounterChannelFromLiveOwnerConnects) {
    ScopedSegment segment{unique_segment_name("ack")};
    InterProcessCounterChannel owner(segment.name);

    auto connector = InterProcessCounterChannel::connect(segment.name, 1000);
    EXPECT_TRUE(connector->had_clean_prior_shutdown());
    owner.inject(3);
    EXPECT_EQ(connector->try_consume_all(), 3u);
}

TEST(ShmOwnerLiveness, CounterChannelWithoutOwnerStampConnects) {
    ScopedSegment segment{unique_segment_name("ack")};
    create_raw_segment(segment.name, /*owner_pid=*/0, /*owner_start_time=*/0);

    auto connector = InterProcessCounterChannel::connect(segment.name, 1000);
    EXPECT_TRUE(connector->had_clean_prior_shutdown());
}

TEST(ShmOwnerLiveness, CounterChannelUnsizedSegmentIsNotExported) {
    ScopedSegment segment{unique_segment_name("ack")};
    // An owner between shm_open(O_CREAT|O_EXCL) and its initialising write: the name exists, the segment is 0 bytes.
    const int fd = ::shm_open(segment.name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
    ASSERT_NE(fd, -1);
    ::close(fd);

    expect_throws_containing(
        [&] { InterProcessCounterChannel::connect(segment.name, 50); }, "timed out after 50 ms waiting for");
}

TEST(ShmOwnerLiveness, CounterChannelFromDeadOwnerIsNotExported) {
    ScopedSegment segment{unique_segment_name("ack")};
    create_raw_segment(segment.name, static_cast<uint32_t>(dead_pid()), /*owner_start_time=*/0);

    expect_throws_containing(
        [&] { InterProcessCounterChannel::connect(segment.name, 50); }, "timed out after 50 ms waiting for");
}

TEST(ShmOwnerLiveness, CounterChannelFromReusedPidIsNotExported) {
    ScopedSegment segment{unique_segment_name("ack")};
    create_raw_segment(segment.name, static_cast<uint32_t>(getpid()), process_start_time(getpid()) + 1);

    expect_throws_containing(
        [&] { InterProcessCounterChannel::connect(segment.name, 50); }, "timed out after 50 ms waiting for");
}

TEST(ShmOwnerLiveness, CounterChannelRecreatedByLiveOwnerIsPickedUp) {
    ScopedSegment segment{unique_segment_name("ack")};
    create_raw_segment(segment.name, static_cast<uint32_t>(dead_pid()), /*owner_start_time=*/0);

    // The successor does what a restarted owner does: unlink the stale segment and create its own.
    std::unique_ptr<InterProcessCounterChannel> owner;
    std::thread successor([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ::shm_unlink(segment.name.c_str());
        owner = std::make_unique<InterProcessCounterChannel>(segment.name);
    });
    auto connector = InterProcessCounterChannel::connect(segment.name, 5000);
    successor.join();

    EXPECT_TRUE(connector->had_clean_prior_shutdown());
    owner->inject(1);
    EXPECT_EQ(connector->try_consume_all(), 1u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
