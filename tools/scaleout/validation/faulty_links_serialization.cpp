// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tools/scaleout/validation/faulty_links_serialization.hpp"
#include "protobuf/faulty_links.pb.h"

namespace tt::scaleout::validation {

std::vector<uint8_t> serialize_faulty_links_to_bytes(const std::vector<::FaultyLink>& faulty_links) {
    FaultyLinksList proto_list;

    for (const auto& faulty_link : faulty_links) {
        auto* proto_faulty_link = proto_list.add_faulty_links();

        // Serialize EthChannelIdentifier
        auto* proto_channel_id = proto_faulty_link->mutable_channel_identifier();
        proto_channel_id->set_host(faulty_link.channel_identifier.host);
        proto_channel_id->set_asic_id(*faulty_link.channel_identifier.asic_id);
        proto_channel_id->set_tray_id(*faulty_link.channel_identifier.tray_id);
        proto_channel_id->set_asic_location(*faulty_link.channel_identifier.asic_location);
        proto_channel_id->set_channel(faulty_link.channel_identifier.channel);

        // Serialize LinkFailure
        auto* proto_link_failure = proto_faulty_link->mutable_link_failure();

        // Serialize RetrainFailure
        auto* proto_retrain = proto_link_failure->mutable_retrain_failure();
        proto_retrain->set_retrain_count(faulty_link.link_failure.retrain_failure.retrain_count);
        auto* proto_retrain_params = proto_retrain->mutable_test_params();
        proto_retrain_params->set_packet_size_bytes(
            faulty_link.link_failure.retrain_failure.test_params.packet_size_bytes);
        proto_retrain_params->set_data_size(faulty_link.link_failure.retrain_failure.test_params.data_size);

        // Serialize CrcErrorFailure
        auto* proto_crc = proto_link_failure->mutable_crc_error_failure();
        proto_crc->set_crc_error_count(faulty_link.link_failure.crc_error_failure.crc_error_count);
        auto* proto_crc_params = proto_crc->mutable_test_params();
        proto_crc_params->set_packet_size_bytes(
            faulty_link.link_failure.crc_error_failure.test_params.packet_size_bytes);
        proto_crc_params->set_data_size(faulty_link.link_failure.crc_error_failure.test_params.data_size);

        // Serialize UncorrectedCodewordFailure
        auto* proto_uncorr = proto_link_failure->mutable_uncorrected_codeword_failure();
        proto_uncorr->set_uncorrected_codeword_count(
            faulty_link.link_failure.uncorrected_codeword_failure.uncorrected_codeword_count);
        auto* proto_uncorr_params = proto_uncorr->mutable_test_params();
        proto_uncorr_params->set_packet_size_bytes(
            faulty_link.link_failure.uncorrected_codeword_failure.test_params.packet_size_bytes);
        proto_uncorr_params->set_data_size(faulty_link.link_failure.uncorrected_codeword_failure.test_params.data_size);

        // Serialize DataMismatchFailure
        auto* proto_data_mismatch = proto_link_failure->mutable_data_mismatch_failure();
        proto_data_mismatch->set_num_mismatched_words(
            faulty_link.link_failure.data_mismatch_failure.num_mismatched_words);
        auto* proto_data_params = proto_data_mismatch->mutable_test_params();
        proto_data_params->set_packet_size_bytes(
            faulty_link.link_failure.data_mismatch_failure.test_params.packet_size_bytes);
        proto_data_params->set_data_size(faulty_link.link_failure.data_mismatch_failure.test_params.data_size);
    }

    // Serialize to bytes
    size_t size = proto_list.ByteSizeLong();
    std::vector<uint8_t> result(size);

    if (!proto_list.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize FaultyLinksList to protobuf binary format");
    }

    return result;
}

std::vector<::FaultyLink> deserialize_faulty_links_from_bytes(const std::vector<uint8_t>& data) {
    FaultyLinksList proto_list;

    if (!proto_list.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse FaultyLinksList from protobuf binary format");
    }

    std::vector<::FaultyLink> faulty_links;
    faulty_links.reserve(proto_list.faulty_links_size());

    for (const auto& proto_faulty_link : proto_list.faulty_links()) {
        ::FaultyLink faulty_link;

        // Deserialize EthChannelIdentifier
        const auto& proto_channel_id = proto_faulty_link.channel_identifier();
        faulty_link.channel_identifier.host = proto_channel_id.host();
        faulty_link.channel_identifier.asic_id = tt::tt_metal::AsicID(proto_channel_id.asic_id());
        faulty_link.channel_identifier.tray_id = tt::tt_metal::TrayID(proto_channel_id.tray_id());
        faulty_link.channel_identifier.asic_location = tt::tt_metal::ASICLocation(proto_channel_id.asic_location());
        faulty_link.channel_identifier.channel = static_cast<uint8_t>(proto_channel_id.channel());

        // Deserialize LinkFailure
        const auto& proto_link_failure = proto_faulty_link.link_failure();

        // Deserialize RetrainFailure
        const auto& proto_retrain = proto_link_failure.retrain_failure();
        faulty_link.link_failure.retrain_failure.retrain_count = proto_retrain.retrain_count();
        faulty_link.link_failure.retrain_failure.test_params.packet_size_bytes =
            proto_retrain.test_params().packet_size_bytes();
        faulty_link.link_failure.retrain_failure.test_params.data_size = proto_retrain.test_params().data_size();

        // Deserialize CrcErrorFailure
        const auto& proto_crc = proto_link_failure.crc_error_failure();
        faulty_link.link_failure.crc_error_failure.crc_error_count = proto_crc.crc_error_count();
        faulty_link.link_failure.crc_error_failure.test_params.packet_size_bytes =
            proto_crc.test_params().packet_size_bytes();
        faulty_link.link_failure.crc_error_failure.test_params.data_size = proto_crc.test_params().data_size();

        // Deserialize UncorrectedCodewordFailure
        const auto& proto_uncorr = proto_link_failure.uncorrected_codeword_failure();
        faulty_link.link_failure.uncorrected_codeword_failure.uncorrected_codeword_count =
            proto_uncorr.uncorrected_codeword_count();
        faulty_link.link_failure.uncorrected_codeword_failure.test_params.packet_size_bytes =
            proto_uncorr.test_params().packet_size_bytes();
        faulty_link.link_failure.uncorrected_codeword_failure.test_params.data_size =
            proto_uncorr.test_params().data_size();

        // Deserialize DataMismatchFailure
        const auto& proto_data_mismatch = proto_link_failure.data_mismatch_failure();
        faulty_link.link_failure.data_mismatch_failure.num_mismatched_words =
            proto_data_mismatch.num_mismatched_words();
        faulty_link.link_failure.data_mismatch_failure.test_params.packet_size_bytes =
            proto_data_mismatch.test_params().packet_size_bytes();
        faulty_link.link_failure.data_mismatch_failure.test_params.data_size =
            proto_data_mismatch.test_params().data_size();

        faulty_links.push_back(std::move(faulty_link));
    }

    return faulty_links;
}

}  // namespace tt::scaleout::validation
