#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Script to apply allocation tracking integration to TT-Metal allocator

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOCATOR_CPP="$SCRIPT_DIR/allocator.cpp"

echo "🔧 Applying Allocation Tracking Integration..."
echo ""

# Check if already applied
if grep -q "allocation_client.hpp" "$ALLOCATOR_CPP"; then
    echo "⚠️  Integration already applied!"
    echo "   If you want to reapply, please revert the changes first."
    exit 1
fi

# Backup original
cp "$ALLOCATOR_CPP" "$ALLOCATOR_CPP.backup"
echo "✓ Created backup: allocator.cpp.backup"

# Apply changes using sed
echo "✓ Adding device.hpp include (for IDevice::id())..."
sed -i '/#include <buffer.hpp>/a\#include <device.hpp>' "$ALLOCATOR_CPP"

echo "✓ Adding allocation_client.hpp include..."
sed -i '/^#include <umd\/device\/types\/xy_pair.hpp>$/a\\n\/\/ NEW: Allocation tracking support\n#include "allocation_client.hpp"' "$ALLOCATOR_CPP"

# Add tracking to allocate_buffer - after allocated_buffers_.insert(buffer);
echo "✓ Instrumenting allocate_buffer()..."
sed -i '/allocated_buffers_\.insert(buffer);$/a\\n    \/\/ NEW: Report allocation to tracking server\n    if (AllocationClient::is_enabled()) {\n        AllocationClient::report_allocation(\n            buffer->device()->id(),\n            size,\n            static_cast<uint8_t>(buffer_type),\n            address\n        );\n    }' "$ALLOCATOR_CPP"

# Add tracking to deallocate_buffer - after getting address but before deallocation
echo "✓ Instrumenting deallocate_buffer()..."
sed -i '/auto buffer_type = buffer->buffer_type();$/a\\n    \/\/ NEW: Report deallocation to tracking server\n    if (AllocationClient::is_enabled()) {\n        AllocationClient::report_deallocation(address);\n    }' "$ALLOCATOR_CPP"

echo ""
echo "✅ Integration applied successfully!"
echo ""
echo "Next steps:"
echo "  1. Rebuild TT-Metal: cmake --build build --target metalium -j"
echo "  2. Start tracking server: ./allocation_server_poc &"
echo "  3. Enable tracking: export TT_ALLOC_TRACKING_ENABLED=1"
echo "  4. Run your application"
echo "  5. Monitor: ./allocation_monitor_client -r 500"
echo ""
echo "To revert changes:"
echo "  mv allocator.cpp.backup allocator.cpp"
