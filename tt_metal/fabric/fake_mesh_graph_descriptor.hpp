#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <protobuf/mesh_graph_descriptor.pb.h>

namespace tt::tt_fabric {

void fake_mesh_graph_descriptor(const std::string& mesh_graph_desc_file_path) {
    proto::MeshGraphDescriptor proto_mgd;
    google::protobuf::TextFormat::Parser parser;

    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);

    TT_FATAL(parser.ParseFromString(mesh_graph_desc_file_path, &proto_mgd), "Failed to parse mesh graph descriptor");
}

}  // namespace tt::tt_fabric
