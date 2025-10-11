// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include "tools/scaleout/validation/utils/ethernet_link_metrics_serialization.hpp"
#include "protobuf/ethernet_link_metrics.pb.h"

namespace tt::scaleout::validation {

std::vector<uint8_t> serialize_link_metrics_to_bytes(const std::vector<::EthernetLinkMetrics>& link_metrics) {
    EthernetLinkMetricsList proto_list;

    for (const auto& link_metric : link_metrics) {
        auto* proto_link_metrics = proto_list.add_link_metrics();

        // Serialize EthChannelIdentifier
        auto* proto_channel_id = proto_link_metrics->mutable_channel_identifier();
        proto_channel_id->set_host(link_metric.channel_identifier.host);
        proto_channel_id->set_asic_id(*link_metric.channel_identifier.asic_id);
        proto_channel_id->set_tray_id(*link_metric.channel_identifier.tray_id);
        proto_channel_id->set_asic_location(*link_metric.channel_identifier.asic_location);
        proto_channel_id->set_channel(link_metric.channel_identifier.channel);

        // Serialize LinkStatus
        auto* proto_link_status = proto_link_metrics->mutable_link_status();

        // Serialize EthernetMetrics
        auto* proto_metrics = proto_link_status->mutable_ethernet_metrics();
        proto_metrics->set_retrain_count(link_metric.link_status.metrics.retrain_count);
        proto_metrics->set_crc_error_count(link_metric.link_status.metrics.crc_error_count);
        proto_metrics->set_corrected_codeword_count(link_metric.link_status.metrics.corrected_codeword_count);
        proto_metrics->set_uncorrected_codeword_count(link_metric.link_status.metrics.uncorrected_codeword_count);

        // Serialize TrafficParams
        auto* proto_traffic_params = proto_link_status->mutable_traffic_params();
        proto_traffic_params->set_packet_size_bytes(link_metric.link_status.traffic_params.packet_size_bytes);
        proto_traffic_params->set_data_size(link_metric.link_status.traffic_params.data_size);

        // Serialize num_mismatched_words
        proto_link_status->set_num_mismatched_words(link_metric.link_status.num_mismatched_words);
    }

    // Serialize to bytes
    size_t size = proto_list.ByteSizeLong();
    std::vector<uint8_t> result(size);

    if (!proto_list.SerializeToArray(result.data(), size)) {
        throw std::runtime_error("Failed to serialize EthernetLinkMetricsList to protobuf binary format");
    }

    return result;
}

std::vector<::EthernetLinkMetrics> deserialize_link_metrics_from_bytes(const std::vector<uint8_t>& data) {
    EthernetLinkMetricsList proto_list;

    if (!proto_list.ParseFromArray(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse EthernetLinkMetricsList from protobuf binary format");
    }

    std::vector<::EthernetLinkMetrics> link_metrics;
    link_metrics.reserve(proto_list.link_metrics_size());

    for (const auto& proto_link_metric : proto_list.link_metrics()) {
        ::EthernetLinkMetrics link_metric;

        // Deserialize EthChannelIdentifier
        const auto& proto_channel_id = proto_link_metric.channel_identifier();
        link_metric.channel_identifier.host = proto_channel_id.host();
        link_metric.channel_identifier.asic_id = tt::tt_metal::AsicID(proto_channel_id.asic_id());
        link_metric.channel_identifier.tray_id = tt::tt_metal::TrayID(proto_channel_id.tray_id());
        link_metric.channel_identifier.asic_location = tt::tt_metal::ASICLocation(proto_channel_id.asic_location());
        link_metric.channel_identifier.channel = static_cast<uint8_t>(proto_channel_id.channel());

        // Deserialize LinkStatus
        const auto& proto_link_status = proto_link_metric.link_status();

        // Deserialize EthernetMetrics
        const auto& proto_metrics = proto_link_status.ethernet_metrics();
        link_metric.link_status.metrics.retrain_count = proto_metrics.retrain_count();
        link_metric.link_status.metrics.crc_error_count = proto_metrics.crc_error_count();
        link_metric.link_status.metrics.corrected_codeword_count = proto_metrics.corrected_codeword_count();
        link_metric.link_status.metrics.uncorrected_codeword_count = proto_metrics.uncorrected_codeword_count();

        // Deserialize TrafficParams
        const auto& proto_traffic_params = proto_link_status.traffic_params();
        link_metric.link_status.traffic_params.packet_size_bytes = proto_traffic_params.packet_size_bytes();
        link_metric.link_status.traffic_params.data_size = proto_traffic_params.data_size();

        // Deserialize num_mismatched_words
        link_metric.link_status.num_mismatched_words = proto_link_status.num_mismatched_words();

        link_metrics.push_back(std::move(link_metric));
    }

    return link_metrics;
}

}  // namespace tt::scaleout::validation
