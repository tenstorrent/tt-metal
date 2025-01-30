
# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=20.04
FROM public.ecr.aws/ubuntu/ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode runtime
