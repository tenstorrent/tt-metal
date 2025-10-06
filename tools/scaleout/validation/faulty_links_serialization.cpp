// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tools/scaleout/validation/faulty_links_serialization.hpp"
#include "protobuf/faulty_links.pb.h"

namespace tt::scaleout::validation {

std::vector<uint8_t> serialize_link_metrics_to_bytes(const std::vector<::EthernetLinkMetrics>& link_metrics) {
    FaultyLinksList proto_list;

    for (const auto& link_metric : link_metrics) {
        auto* proto_faulty_link = proto_list.add_faulty_links();

        // Serialize EthChannelIdentifier
        auto* proto_channel_id = proto_faulty_link->mutable_channel_identifier();
        proto_channel_id->set_host(link_metric.channel_identifier.host);
        proto_channel_id->set_asic_id(*link_metric.channel_identifier.asic_id);
        proto_channel_id->set_tray_id(*link_metric.channel_identifier.tray_id);
        proto_channel_id->set_asic_location(*link_metric.channel_identifier.asic_location);
        proto_channel_id->set_channel(link_metric.channel_identifier.channel);

        // Serialize LinkStatus - now much simpler!
        auto* proto_link_failure = proto_faulty_link->mutable_link_failure();

        // Serialize test params once
        auto* proto_retrain = proto_link_failure->mutable_retrain_failure();
        auto* proto_test_params = proto_retrain->mutable_test_params();
        proto_test_params->set_packet_size_bytes(link_metric.link_status.test_params.packet_size_bytes);
        proto_test_params->set_data_size(link_metric.link_status.test_params.data_size);

        // Serialize metrics
        proto_retrain->set_retrain_count(link_metric.link_status.metrics.retrain_count);

        auto* proto_crc = proto_link_failure->mutable_crc_error_failure();
        proto_crc->set_crc_error_count(link_metric.link_status.metrics.crc_error_count);
        proto_crc->mutable_test_params()->CopyFrom(*proto_test_params);

        auto* proto_uncorr = proto_link_failure->mutable_uncorrected_codeword_failure();
        proto_uncorr->set_uncorrected_codeword_count(link_metric.link_status.metrics.uncorrected_codeword_count);
        proto_uncorr->mutable_test_params()->CopyFrom(*proto_test_params);

        auto* proto_data_mismatch = proto_link_failure->mutable_data_mismatch_failure();
        proto_data_mismatch->set_num_mismatched_words(link_metric.link_status.num_mismatched_words);
        proto_data_mismatch->mutable_test_params()->CopyFrom(*proto_test_params);
    }

    // Serialize to bytes
    size_t size = proto_list.ByteSizeLong();
    std::vector<uint8_t> result(size);

    if (!proto_list.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize FaultyLinksList to protobuf binary format");
    }

    return result;
}

std::vector<::EthernetLinkMetrics> deserialize_link_metrics_from_bytes(const std::vector<uint8_t>& data) {
    FaultyLinksList proto_list;

    if (!proto_list.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse FaultyLinksList from protobuf binary format");
    }

    std::vector<::EthernetLinkMetrics> link_metrics;
    link_metrics.reserve(proto_list.faulty_links_size());

    for (const auto& proto_faulty_link : proto_list.faulty_links()) {
        ::EthernetLinkMetrics link_metric;

        // Deserialize EthChannelIdentifier
        const auto& proto_channel_id = proto_faulty_link.channel_identifier();
        link_metric.channel_identifier.host = proto_channel_id.host();
        link_metric.channel_identifier.asic_id = tt::tt_metal::AsicID(proto_channel_id.asic_id());
        link_metric.channel_identifier.tray_id = tt::tt_metal::TrayID(proto_channel_id.tray_id());
        link_metric.channel_identifier.asic_location = tt::tt_metal::ASICLocation(proto_channel_id.asic_location());
        link_metric.channel_identifier.channel = static_cast<uint8_t>(proto_channel_id.channel());

        // Deserialize LinkStatus - now much simpler!
        const auto& proto_link_failure = proto_faulty_link.link_failure();
        const auto& proto_retrain = proto_link_failure.retrain_failure();

        // Deserialize test params once
        link_metric.link_status.test_params.packet_size_bytes = proto_retrain.test_params().packet_size_bytes();
        link_metric.link_status.test_params.data_size = proto_retrain.test_params().data_size();

        // Deserialize metrics
        link_metric.link_status.metrics.retrain_count = proto_retrain.retrain_count();
        link_metric.link_status.metrics.crc_error_count = proto_link_failure.crc_error_failure().crc_error_count();
        link_metric.link_status.metrics.uncorrected_codeword_count =
            proto_link_failure.uncorrected_codeword_failure().uncorrected_codeword_count();
        link_metric.link_status.num_mismatched_words =
            proto_link_failure.data_mismatch_failure().num_mismatched_words();

        link_metrics.push_back(std::move(link_metric));
    }

    return link_metrics;
}

}  // namespace tt::scaleout::validation
