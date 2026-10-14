// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Runtime CLI for the X280 programming component.
//
// Opens a unit MeshDevice, then initializes X280 through X280Runtime::open().
// That order is the contract: the component binds to a mesh tt-metal already
// owns and never creates a device context of its own. Runtime initialization
// may automatically prime LIM and return reset-required without a runtime.
//
// Pinned-input operations use the active-image lifecycle for native-width mask
// register access.
//
//   x280_runtime_cli                            initialize, inventory, and inspect
//   x280_runtime_cli --tile 0 --pll 1000 --boot boot, then set the operating PLL
//
// Exit codes: 0 success, 4 an ASIC reset is required, 1 anything else.

#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_tile.hpp"
#include "rt/x280_pinned_input.h"

#include <mesh_device.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailed = 1;
constexpr int kExitResetRequired = 4;

constexpr const char* kImageEnvVar = "X280_IDLE_FW_IMAGE";
// The resident-idle image this build's x280_images target produces.
constexpr const char* kDefaultIdleImage = X280_IMAGE_DIR "/firmware-lim-idle.bin";
constexpr const char* kPinnedInputPrepareImageEnvVar = "X280_PINNED_INPUT_PREPARE_FW_IMAGE";
constexpr const char* kPinnedInputTeardownImageEnvVar = "X280_PINNED_INPUT_TEARDOWN_FW_IMAGE";

// Operating frequency applied after boot, not the safe reset-release clock,
// which boot establishes itself.
constexpr uint32_t kDefaultPllMhz = 1000;

enum class PinnedInputOperation {
    None,
    Prepare,
    Teardown,
};

struct Options {
    int chip = 0;
    // Hardware L2CPU indices, in the order given. More than one selects the
    // chip-level batch boot or sequential pinned-input operation.
    std::vector<int> tiles{0};
    bool all_tiles = false;
    uint8_t lim_ways = x280::kMaxLimWays;
    bool lim_ways_specified = false;
    uint32_t pll_mhz = kDefaultPllMhz;
    bool pll_specified = false;
    int timeout_ms = 3000;
    bool boot = false;
    bool dry_run = false;
    bool require_prepared = false;
    bool exclusive_device = false;
    bool fresh_cache = false;
    PinnedInputOperation pinned_input = PinnedInputOperation::None;
    bool idle_image_specified = false;
    std::string image_path;
    std::string pinned_input_prepare_image_path;
    std::string pinned_input_teardown_image_path;
};

std::string describe_access_path(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, int chip_id) {
    if (!mesh) {
        throw std::runtime_error("x280_runtime_cli: access-path diagnostics require a live MeshDevice");
    }

    const auto chip_ids = mesh->get_device_ids();
    if (std::find(chip_ids.begin(), chip_ids.end(), chip_id) == chip_ids.end()) {
        throw std::runtime_error(
            "x280_runtime_cli: chip " + std::to_string(chip_id) + " is outside the caller's MeshDevice");
    }
    return "chip " + std::to_string(chip_id) +
           " through the runtime-shared Metal transport bound to the caller's MeshDevice";
}

void usage(const char* argv0) {
    std::printf(
        "usage: %s [--chip N] [--tile N[,N...]] [--all-tiles] [--lim-ways N]\n"
        "       [--pll MHZ] [--timeout MS]\n"
        "       [--boot] [--dry-run] [--require-prepared] [--idle-image PATH]\n"
        "       [--prepare-pinned-input --fresh-cache --exclusive-device]\n"
        "       [--teardown-pinned-input --exclusive-device]\n"
        "       [--prepare-pinned-input-image PATH]\n"
        "       [--teardown-pinned-input-image PATH]\n"
        "\n"
        "  Runtime open automatically initializes chip PLL/ECC state. On an\n"
        "  unprimed chip it commits WayEnable, exits 4, and requires an ASIC\n"
        "  reset before the next invocation can return a runtime. Pinned-input\n"
        "  operations instead force RequirePrepared + RequireExact at 14 LIM\n"
        "  ways and perform no priming or shape application.\n"
        "\n"
        "  --require-prepared\n"
        "                  bind only to already-primed hardware. Open fails\n"
        "                  instead of committing WayEnable, so no invocation can\n"
        "                  cost an ASIC reset as a side effect.\n"
        "  --tile          one L2CPU index, or several comma-separated. More than\n"
        "                  one boots them as a batch: their resets are released in\n"
        "                  a single write, so no tile runs while another leaves\n"
        "                  reset. One batch per chip-reset lifetime.\n"
        "  --all-tiles     select every enabled, unharvested L2CPU of the chip.\n"
        "  --lim-ways      request N backed 128 KiB LIM ways, in [0, 15]. The\n"
        "                  default is 15. This uses the runtime's increase-only\n"
        "                  ApplyWhileReset shape path and requires an explicit\n"
        "                  --require-prepared so it cannot hide ECC priming.\n"
        "                  Pinned-input operations have a fixed 14-way profile\n"
        "                  and reject this option.\n"
        "  --boot          allow releasing L2CPU reset to cold boot resident idle.\n"
        "                  One-way for the chip-reset lifetime, so it is opt-in.\n"
        "                  Without it, the ready runtime's tile is only inspected.\n"
        "  --pll           operating frequency, applied after a successful boot\n"
        "                  (default %u MHz). Boot establishes the safe clock on\n"
        "                  its own; this retunes the chip afterwards, with the\n"
        "                  resident idle already running. PLL4 is chip-global,\n"
        "                  so it moves every L2CPU on the chip.\n"
        "  --dry-run       do everything a cold boot does except release reset:\n"
        "                  load and verify the image, write and verify the reset\n"
        "                  vectors, clear the mailboxes. Nothing becomes one-way.\n"
        "  --idle-image    path to the resident-idle image. Falls back to $%s,\n"
        "                  then to %s,\n"
        "                  which ninja x280_images builds.\n"
        "  --prepare-pinned-input\n"
        "                  admit exact WAYENABLE=1 and baseline mask bits 11,\n"
        "                  populate the fixed 4288-byte Zero Device region via\n"
        "                  hart 0, then pin way 1 with mask bits 01 for all 38\n"
        "                  masters. Requires explicit fresh-cache provenance and\n"
        "                  externally exclusive ownership; performs no reset,\n"
        "                  invalidation, ECC priming, or WAYENABLE change.\n"
        "  --teardown-pinned-input\n"
        "                  after every selected tile is confirmed in resident\n"
        "                  idle, admit mask bits 01 and re-enable only way 1 to\n"
        "                  return to bits 11. Performs no flush or WAYENABLE\n"
        "                  change. This is normal final teardown, not recovery\n"
        "                  after a failed prepare, unconfirmed cleanup, or crash.\n"
        "  --exclusive-device\n"
        "                  assert that external operator/device reservation owns\n"
        "                  the selected tiles across prepare, benchmark\n"
        "                  executables, and final teardown. This process cannot\n"
        "                  infer or preserve that ownership after it exits.\n"
        "  --fresh-cache   assert coordinated fresh placement provenance for\n"
        "                  preparation. Bytes and register readback do not prove\n"
        "                  this condition; a failed prepare aborts the epoch.\n"
        "  --prepare-pinned-input-image\n"
        "                  preparation-only active image. Falls back to $%s.\n"
        "                  Build it with: ninja x280_images\n"
        "  --teardown-pinned-input-image\n"
        "                  teardown-only active image. Falls back to $%s.\n"
        "                  Build it with: ninja x280_images\n"
        "\n"
        "exit: 0 success, 4 an ASIC reset is required, 1 failed\n",
        argv0,
        kDefaultPllMhz,
        kImageEnvVar,
        kDefaultIdleImage,
        kPinnedInputPrepareImageEnvVar,
        kPinnedInputTeardownImageEnvVar);
}

// Strict integer parsing, bounded. std::atoi maps a malformed argument to 0
// silently, which here turns `--tile typo --boot` into a one-way reset release
// on tile 0 and `--pll abc` into a frequency the component has no solution for.
// Everything is rejected before the device is opened.
bool parse_bounded(const char* option, const char* text, long low, long high, long& out) {
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || value < low || value > high) {
        std::fprintf(stderr, "%s expects an integer in [%ld, %ld], got: %s\n\n", option, low, high, text);
        return false;
    }
    out = value;
    return true;
}

// Parse comma-separated hardware indices. Duplicates are rejected because
// reset release is one-way.
bool parse_tile_list(const char* text, std::vector<int>& out) {
    std::vector<int> parsed;
    const std::string all(text);
    size_t start = 0;
    while (start <= all.size()) {
        const size_t comma = all.find(',', start);
        const std::string field = all.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        long value = 0;
        // The component's own bound: hardware L2CPU numbering is 0..3.
        if (!parse_bounded("--tile", field.c_str(), 0, x280::kNumL2cpuTiles - 1, value)) {
            return false;
        }
        const int index = static_cast<int>(value);
        if (std::find(parsed.begin(), parsed.end(), index) != parsed.end()) {
            std::fprintf(stderr, "--tile lists L2CPU %d more than once\n\n", index);
            return false;
        }
        parsed.push_back(index);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (parsed.empty()) {
        std::fprintf(stderr, "--tile expects at least one L2CPU index\n\n");
        return false;
    }
    out = std::move(parsed);
    return true;
}

// Returns false on a malformed argument, having already explained it.
bool parse(int argc, char** argv, Options& options) {
    long value = 0;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_value = (i + 1) < argc;

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(kExitOk);
        } else if (arg == "--boot") {
            options.boot = true;
        } else if (arg == "--dry-run") {
            options.dry_run = true;
        } else if (arg == "--require-prepared") {
            options.require_prepared = true;
        } else if (arg == "--prepare-pinned-input") {
            if (options.pinned_input != PinnedInputOperation::None) {
                std::fprintf(stderr, "select exactly one pinned-input operation\n\n");
                return false;
            }
            options.pinned_input = PinnedInputOperation::Prepare;
        } else if (arg == "--teardown-pinned-input") {
            if (options.pinned_input != PinnedInputOperation::None) {
                std::fprintf(stderr, "select exactly one pinned-input operation\n\n");
                return false;
            }
            options.pinned_input = PinnedInputOperation::Teardown;
        } else if (arg == "--exclusive-device") {
            options.exclusive_device = true;
        } else if (arg == "--fresh-cache") {
            options.fresh_cache = true;
        } else if (arg == "--chip" && has_value) {
            if (!parse_bounded("--chip", argv[++i], 0, INT_MAX, value)) {
                return false;
            }
            options.chip = static_cast<int>(value);
        } else if (arg == "--tile" && has_value) {
            if (!parse_tile_list(argv[++i], options.tiles)) {
                return false;
            }
        } else if (arg == "--all-tiles") {
            options.all_tiles = true;
        } else if (arg == "--lim-ways" && has_value) {
            if (!parse_bounded("--lim-ways", argv[++i], 0, x280::kMaxLimWays, value)) {
                return false;
            }
            options.lim_ways = static_cast<uint8_t>(value);
            options.lim_ways_specified = true;
        } else if (arg == "--pll" && has_value) {
            // Only a sanity range here; the component owns the solution table
            // and rejects an unsupported frequency before it programs anything.
            if (!parse_bounded("--pll", argv[++i], 1, INT_MAX, value)) {
                return false;
            }
            options.pll_mhz = static_cast<uint32_t>(value);
            options.pll_specified = true;
        } else if (arg == "--timeout" && has_value) {
            if (!parse_bounded("--timeout", argv[++i], 1, INT_MAX, value)) {
                return false;
            }
            options.timeout_ms = static_cast<int>(value);
        } else if (arg == "--idle-image" && has_value) {
            options.image_path = argv[++i];
            options.idle_image_specified = true;
        } else if (arg == "--prepare-pinned-input-image" && has_value) {
            options.pinned_input_prepare_image_path = argv[++i];
        } else if (arg == "--teardown-pinned-input-image" && has_value) {
            options.pinned_input_teardown_image_path = argv[++i];
        } else {
            std::fprintf(stderr, "unrecognised or incomplete argument: %s\n\n", arg.c_str());
            usage(argv[0]);
            return false;
        }
    }

    if (options.image_path.empty()) {
        // The component hard-codes no build-tree location; this driver defaults to its own build's.
        const char* from_env = std::getenv(kImageEnvVar);
        options.image_path = (from_env != nullptr && *from_env != '\0') ? from_env : kDefaultIdleImage;
    }
    if (options.lim_ways_specified && options.pinned_input != PinnedInputOperation::None) {
        std::fprintf(
            stderr, "pinned-input operations use the fixed 14-LIM-way exact profile and do not accept --lim-ways\n\n");
        return false;
    }
    if (options.lim_ways_specified && !options.require_prepared) {
        std::fprintf(
            stderr,
            "--lim-ways requires --require-prepared so shape provisioning cannot also trigger automatic ECC "
            "priming\n\n");
        return false;
    }
    if (options.pinned_input != PinnedInputOperation::None) {
        const bool prepare = options.pinned_input == PinnedInputOperation::Prepare;
        std::string& selected_image =
            prepare ? options.pinned_input_prepare_image_path : options.pinned_input_teardown_image_path;
        const std::string& other_image =
            prepare ? options.pinned_input_teardown_image_path : options.pinned_input_prepare_image_path;
        const char* selected_option = prepare ? "--prepare-pinned-input-image" : "--teardown-pinned-input-image";
        const char* other_option = prepare ? "--teardown-pinned-input-image" : "--prepare-pinned-input-image";
        const char* selected_env = prepare ? kPinnedInputPrepareImageEnvVar : kPinnedInputTeardownImageEnvVar;

        if (!other_image.empty()) {
            std::fprintf(
                stderr, "%s cannot be used with --%s-pinned-input\n\n", other_option, prepare ? "prepare" : "teardown");
            return false;
        }
        if (selected_image.empty()) {
            if (const char* from_env = std::getenv(selected_env); from_env != nullptr && *from_env != '\0') {
                selected_image = from_env;
            }
        }
        if (options.boot || options.dry_run) {
            std::fprintf(stderr, "pinned-input operations do not boot, release reset, or run a boot dry-run\n\n");
            return false;
        }
        if (options.pll_specified || options.idle_image_specified) {
            std::fprintf(
                stderr,
                "pinned-input operations do not accept --pll or --idle-image; they neither move PLL4 nor boot "
                "resident idle\n\n");
            return false;
        }
        if (!options.exclusive_device) {
            std::fprintf(
                stderr,
                "pinned-input operations require --exclusive-device; process-local Metal ownership is not the "
                "suite reservation\n\n");
            return false;
        }
        if (options.pinned_input == PinnedInputOperation::Prepare && !options.fresh_cache) {
            std::fprintf(
                stderr,
                "--prepare-pinned-input requires --fresh-cache; register or content readback cannot establish "
                "fresh placement\n\n");
            return false;
        }
        if (options.pinned_input == PinnedInputOperation::Teardown && options.fresh_cache) {
            std::fprintf(stderr, "--fresh-cache applies only to --prepare-pinned-input\n\n");
            return false;
        }
        if (selected_image.empty()) {
            std::fprintf(
                stderr,
                "no %s firmware image: pass %s PATH or set %s\n",
                prepare ? "pinned-input preparation" : "pinned-input teardown",
                selected_option,
                selected_env);
            return false;
        }
    } else if (
        options.exclusive_device || options.fresh_cache || !options.pinned_input_prepare_image_path.empty() ||
        !options.pinned_input_teardown_image_path.empty()) {
        std::fprintf(
            stderr,
            "--exclusive-device, --fresh-cache, and pinned-input image options require a pinned-input operation\n\n");
        return false;
    }
    return true;
}

const char* pinned_input_operation_name(PinnedInputOperation operation) {
    return operation == PinnedInputOperation::Prepare ? "prepare" : "teardown";
}

bool selected_tiles_idle(x280::X280Runtime& runtime, const std::vector<x280::TileId>& selection) {
    if (!runtime.sessions().empty()) {
        std::fprintf(stderr, "pinned-input operation refused: this runtime still owns active sessions\n");
        return false;
    }
    for (const auto& id : selection) {
        auto& tile = runtime.tile(id);
        const auto state = tile.inspect();
        if (!state.observation_valid || !state.resident_idle_live || !tile.healthy()) {
            std::fprintf(
                stderr,
                "pinned-input operation refused: chip %d L2CPU %d is not confirmed healthy resident idle\n%s",
                id.chip_id,
                id.hw_index,
                x280::describe(state).c_str());
            return false;
        }
    }
    return true;
}

bool run_pinned_input_operation(x280::X280Runtime& runtime, const x280::TileId& id, const Options& options) {
    auto& tile = runtime.tile(id);
    const uint64_t run_value = UINT64_C(0x50494E0000000000) | (static_cast<uint64_t>(options.pinned_input) << 32) |
                               static_cast<uint64_t>(id.hw_index + 1);

    x280::ActiveRunRequest request;
    request.image_path = options.pinned_input == PinnedInputOperation::Prepare
                             ? options.pinned_input_prepare_image_path
                             : options.pinned_input_teardown_image_path;
    request.run_value = run_value;
    request.cleanup_timeout = std::chrono::milliseconds(options.timeout_ms);

    const auto enqueued = runtime.enqueue(tile, request);
    if (!enqueued.enqueued()) {
        std::fprintf(
            stderr,
            "chip %d L2CPU %d pinned-input %s enqueue failed: %s\n",
            id.chip_id,
            id.hw_index,
            pinned_input_operation_name(options.pinned_input),
            x280::describe(enqueued).c_str());
        if (enqueued.session) {
            const auto cleanup = runtime.cleanup(enqueued.session, request.cleanup_timeout);
            std::fprintf(stderr, "cleanup after uncertain enqueue: %s\n", x280::describe(cleanup).c_str());
        }
        return false;
    }

    const auto wait =
        runtime.session(enqueued.session).wait_until_result_ready(std::chrono::milliseconds(options.timeout_ms));
    bool firmware_ok = wait.outcome == x280::WaitOutcome::ResultReady;
    const uint64_t expected_hart0 = options.pinned_input == PinnedInputOperation::Prepare
                                        ? X280_PINNED_INPUT_RESULT_PREPARED
                                        : X280_PINNED_INPUT_RESULT_TORN_DOWN;
    for (uint32_t hart = 0; hart < x280::kHartsPerTile; ++hart) {
        const uint64_t expected = hart == 0 ? expected_hart0 : X280_PINNED_INPUT_RESULT_NOOP_HART;
        const uint64_t observed = wait.hartstates[hart].state.result;
        std::printf(
            "chip %d L2CPU %d hart %u pinned-input result: 0x%016llx\n",
            id.chip_id,
            id.hw_index,
            hart,
            static_cast<unsigned long long>(observed));
        firmware_ok = firmware_ok && wait.hartstates[hart].observation_valid && observed == expected;
    }

    const auto cleanup = runtime.cleanup(enqueued.session, request.cleanup_timeout);
    const auto final_state = tile.inspect();
    const bool stopped = cleanup.cleaned_up() && final_state.observation_valid && final_state.resident_idle_live;
    if (!firmware_ok || !stopped) {
        std::fprintf(
            stderr,
            "chip %d L2CPU %d pinned-input %s failed; the prepared epoch is aborted and must not be reused\n"
            "wait: %s\ncleanup: %s\nfinal:\n%s",
            id.chip_id,
            id.hw_index,
            pinned_input_operation_name(options.pinned_input),
            x280::describe(wait).c_str(),
            x280::describe(cleanup).c_str(),
            x280::describe(final_state).c_str());
        return false;
    }

    std::printf(
        "chip %d L2CPU %d pinned-input %s completed and all harts returned to resident idle\n",
        id.chip_id,
        id.hw_index,
        pinned_input_operation_name(options.pinned_input));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) {
        return kExitFailed;
    }

    try {
        // tt-metal opens the device; the component binds to it. Doing this the
        // other way round breaks the mesh: touching MetalContext first creates a
        // silicon context, and the later create_unit_mesh then throws.
        auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(options.chip);
        x280::RuntimeOpenConfig open_config{
            .priming_policy = options.require_prepared ? x280::LimEccPrimingPolicy::RequirePrepared
                                                       : x280::LimEccPrimingPolicy::AutoPrime,
            .lim_shape = x280::LimShape{options.lim_ways},
        };
        if (options.pinned_input != PinnedInputOperation::None) {
            open_config.priming_policy = x280::LimEccPrimingPolicy::RequirePrepared;
            open_config.lim_shape = x280::LimShape{x280::kPreparedLimWays};
            open_config.shape_change_policy = x280::LimShapeChangePolicy::RequireExact;
        }
        auto opened = x280::X280Runtime::open(mesh, open_config);
        std::printf("%s\n", describe_access_path(mesh, options.chip).c_str());
        for (const auto& chip : opened.chips) {
            std::printf(
                "runtime open chip %d: %s: %s\n", chip.chip_id, x280::to_string(chip.outcome), chip.diagnostic.c_str());
        }
        if (!opened.ready()) {
            std::fprintf(stderr, "runtime open did not produce a ready runtime: %s\n", opened.diagnostic.c_str());
            return opened.requires_preparation() || opened.requires_reset() ? kExitResetRequired : kExitFailed;
        }
        auto runtime = std::move(*opened.runtime);

        // Identities, not handles: this only prints the inventory.
        const auto tiles = runtime.tile_ids();
        std::printf("discovered %zu enabled, unharvested L2CPU tiles:\n", tiles.size());
        for (const auto& tile : tiles) {
            std::printf(
                "  chip %d L2CPU %d at noc0 (%u,%u), %u harts\n",
                tile.chip_id,
                tile.hw_index,
                tile.noc_x,
                tile.noc_y,
                tile.num_harts);
        }

        std::vector<x280::TileId> selection;
        if (options.all_tiles) {
            for (const auto& id : tiles) {
                if (id.chip_id == options.chip) {
                    selection.push_back(id);
                }
            }
            if (selection.empty()) {
                std::fprintf(stderr, "chip %d has no enabled, unharvested L2CPU tiles\n", options.chip);
                return kExitFailed;
            }
        } else {
            for (int hw_index : options.tiles) {
                const auto found = std::find_if(tiles.begin(), tiles.end(), [&](const x280::TileId& id) {
                    return id.chip_id == options.chip && id.hw_index == hw_index;
                });
                if (found == tiles.end()) {
                    std::fprintf(stderr, "chip %d L2CPU %d is not enabled and unharvested\n", options.chip, hw_index);
                    return kExitFailed;
                }
                selection.push_back(*found);
            }
        }

        if (options.pinned_input != PinnedInputOperation::None) {
            if (!selected_tiles_idle(runtime, selection)) {
                return kExitFailed;
            }
            for (const auto& id : selection) {
                if (!run_pinned_input_operation(runtime, id, options)) {
                    std::fprintf(
                        stderr,
                        "pinned-input %s stopped after the first failure; no automatic rollback or recovery was "
                        "attempted\n",
                        pinned_input_operation_name(options.pinned_input));
                    return kExitFailed;
                }
            }
            if (options.pinned_input == PinnedInputOperation::Prepare) {
                std::printf(
                    "prepared policy remains in hardware after this process closes. Keep the external exclusive "
                    "reservation until explicit final teardown; per-case runtime close does not unpin it.\n");
            } else {
                std::printf(
                    "final teardown re-enabled only reserved way 1. WAYENABLE remains 1 (14 LIM ways), and no cache "
                    "line was flushed.\n");
            }
            return kExitOk;
        }

        if (selection.size() > 1) {
            // Batch preparation is all-or-nothing; --dry-run remains per tile.
            if (options.dry_run) {
                std::fprintf(
                    stderr,
                    "--dry-run covers one tile: it proves the reset-vector path without releasing anything, so run "
                    "it per tile before a batch\n");
                return kExitFailed;
            }

            for (const auto& id : selection) {
                std::printf("\n%s", x280::describe(runtime.tile(id).inspect()).c_str());
            }
            if (!options.boot) {
                std::printf(
                    "\ninspection only after runtime initialization: nothing was booted. "
                    "Pass --boot to allow reset release, which is one-way for this chip-reset lifetime.\n");
                return kExitOk;
            }
            x280::MultiTileBootRequest request;
            request.tiles = selection;
            request.idle_image_path = options.image_path;
            request.boot_timeout = std::chrono::milliseconds(options.timeout_ms);
            request.allow_reset_release = options.boot;

            const auto result = runtime.boot_tiles(request);
            std::printf("\nbatch: %s\n", result.diagnostic.c_str());
            for (const auto& report : result.tiles) {
                std::printf(
                    "  L2CPU %d: %s: %s\n",
                    report.id.hw_index,
                    x280::to_string(report.result.outcome),
                    report.result.diagnostic.c_str());
            }
            if (!result.ready()) {
                return result.reset_released ? kExitResetRequired : kExitFailed;
            }

            runtime.set_pll(options.chip, options.pll_mhz, std::chrono::milliseconds(options.timeout_ms));
            bool every_tile_live = true;
            for (const auto& id : selection) {
                const auto final_state = runtime.tile(id).inspect();
                std::printf("\n%s", x280::describe(final_state).c_str());
                every_tile_live = every_tile_live && final_state.observation_valid && final_state.resident_idle_live;
            }
            return every_tile_live ? kExitOk : kExitFailed;
        }

        auto& tile = runtime.tile(selection.front());
        const auto initial = tile.inspect();

        if (!options.boot && !options.dry_run) {
            std::printf("\n%s", x280::describe(initial).c_str());
            std::printf(
                "\ninspection only after runtime initialization: nothing was booted. "
                "Pass --boot to allow reset release, which is one-way for this chip-reset lifetime.\n");
            return initial.observation_valid ? kExitOk : kExitFailed;
        }

        x280::BootRequest request;
        request.idle_image_path = options.image_path;
        request.boot_timeout = std::chrono::milliseconds(options.timeout_ms);
        request.allow_reset_release = options.boot;
        request.dry_run = options.dry_run;

        const auto result = tile.boot(request);
        std::printf("\noutcome: %s\n%s\n", x280::to_string(result.outcome), result.diagnostic.c_str());
        std::printf("\n%s", x280::describe(result.last_state).c_str());

        if (result.ready()) {
            if (!options.dry_run) {
                runtime.set_pll(options.chip, options.pll_mhz, std::chrono::milliseconds(options.timeout_ms));
            }
            const auto final = tile.inspect();
            std::printf("\n%s", x280::describe(final).c_str());
            return final.observation_valid && final.resident_idle_live ? kExitOk : kExitFailed;
        }

        if (result.outcome == x280::BootOutcome::PreparedResetNotReleased) {
            return kExitOk;
        }
        return x280::reset_required(result.outcome) ? kExitResetRequired : kExitFailed;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "x280_runtime_cli: %s\n", error.what());
        return kExitFailed;
    }
}
