# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Cap'N Proto file schema: https://capnproto.org/language.html

@0xf8b5d6e3c2a19074;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("tt::tt_metal::inspector::rpc");

interface InspectorChannel {
    # Returns name of the channel
    getName @0 () -> (name :Text);

    # Serializes all functions that have no arguments into specified directory
    serializeRpc @1 (path: Text) -> ();
}

interface InspectorChannelRegistry {
    # Get interface for the specified channel
    getChannel @0 (name: Text) -> (channel :InspectorChannel);

    # Get all registered channels
    getChannelNames @1 () -> (names :List(Text));
}
