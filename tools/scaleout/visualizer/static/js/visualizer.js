// Network Cabling Visualizer - Client-side JavaScript

// ===== Configuration Constants =====
const LAYOUT_CONSTANTS = {
    DEFAULT_RACK_WIDTH: 450,
    MIN_START_X: 350,
    MIN_START_Y: 450,
    RACK_X_OFFSET: 100,
    RACK_Y_OFFSET: 150,
    NEW_RACK_DEFAULT_X: 250,
    NEW_RACK_DEFAULT_Y: 300,
    RACK_SPACING_BUFFER: 1.35  // 35% extra space
};

const CONNECTION_COLORS = {
    INTRA_NODE: '#4CAF50',  // Green for same node
    INTER_NODE: '#2196F3'   // Blue for different nodes
};

const DEFAULT_CABLE_CONFIG = {
    type: 'QSFP_DD',
    length: 'Unknown'
};

// ===== Global Variables =====
let cy;
let currentData = null;
let selectedConnection = null;
let isEdgeCreationMode = false;
let sourcePort = null;

// Node configurations - now loaded from server-side to ensure consistency
// This will be populated from window.SERVER_NODE_CONFIGS injected by the server
let NODE_CONFIGS = {};

// Initialize NODE_CONFIGS from server-side data
function initializeNodeConfigs() {
    if (window.SERVER_NODE_CONFIGS && Object.keys(window.SERVER_NODE_CONFIGS).length > 0) {
        NODE_CONFIGS = window.SERVER_NODE_CONFIGS;
        console.log('Node configurations loaded from server:', NODE_CONFIGS);
    } else {
        // Fallback to hardcoded configs if server data is not available
        NODE_CONFIGS = {
            'N300_LB': { tray_count: 4, ports_per_tray: 2, tray_layout: 'horizontal' },
            'N300_QB': { tray_count: 4, ports_per_tray: 2, tray_layout: 'horizontal' },
            'WH_GALAXY': { tray_count: 4, ports_per_tray: 6, tray_layout: 'vertical' },
            'BH_GALAXY': { tray_count: 4, ports_per_tray: 14, tray_layout: 'vertical' },
            'P150_QB_GLOBAL': { tray_count: 4, ports_per_tray: 4, tray_layout: 'horizontal' },
            'P150_QB_AMERICA': { tray_count: 4, ports_per_tray: 4, tray_layout: 'horizontal' }
        };
        console.warn('Using fallback node configurations - server configs not available');
    }
}

function getNextConnectionNumber() {
    if (!cy) return 0;

    // Get all existing edges and find the highest connection number
    const allEdges = cy.edges();
    let maxConnectionNumber = -1;

    allEdges.forEach(edge => {
        const connectionNum = edge.data('connection_number');
        if (typeof connectionNum === 'number' && connectionNum > maxConnectionNumber) {
            maxConnectionNumber = connectionNum;
        }
    });

    // Return the next number (0 if no connections exist, otherwise max + 1)
    return maxConnectionNumber + 1;
}

function getEthChannelMapping(nodeType, portNumber) {
    // Get the node type from 2 levels above the port (shelf level)
    if (!nodeType || !portNumber) return 'Unknown';

    const nodeTypeUpper = nodeType.toUpperCase();

    // Define eth channel mappings based on node type and port number
    switch (nodeTypeUpper) {
        case 'N300_LB':
        case 'N300_QB':
            // N300 nodes: 2 ports per tray, specific channel mapping
            if (portNumber === 1) return 'ASIC: 0 Channel: 6-7';
            if (portNumber === 2) return 'ASIC: 0 Channel: 0-1';
            break;

        case 'WH_GALAXY':
            // WH_GALAXY: 6 ports per tray,
            if (portNumber === 1) return 'ASIC: 5 Channel: 4-7';
            if (portNumber === 2) return 'ASIC: 1 Channel: 4-7';
            if (portNumber === 3) return 'ASIC: 1 Channel: 0-3';
            if (portNumber === 4) return 'ASIC: 2 Channel: 0-3';
            if (portNumber === 5) return 'ASIC: 3 Channel: 0-3';
            if (portNumber === 6) return 'ASIC: 4 Channel: 0-3';
            break;

        case 'BH_GALAXY':
            // BH_GALAXY: 14 ports per tray
            // TODO: seeing discrepancies in the channel mapping for BH_GALAXY. Need to verify with team.
            if (portNumber === 1) return 'ASIC: 5 Channel: 2-3';
            if (portNumber === 2) return 'ASIC: 1 Channel: 2-3';
            if (portNumber === 3) return 'ASIC: 1 Channel: 0-1';
            if (portNumber === 4) return 'ASIC: 2 Channel: 0-1';
            if (portNumber === 5) return 'ASIC: 3 Channel: 0-1';
            if (portNumber === 6) return 'ASIC: 4 Channel: 0-1';
            if (portNumber === 7) return 'ASIC: 1 Channel: 10, ASIC: 2 Channel: 10';
            if (portNumber === 8) return 'ASIC: 5 Channel: 10, ASIC: 6 Channel: 10';
            if (portNumber === 9) return 'ASIC: 3 Channel: 10, ASIC: 4 Channel: 10';
            if (portNumber === 10) return 'ASIC: 7 Channel: 10, ASIC: 8 Channel: 10';
            if (portNumber === 11) return 'ASIC: 1 Channel: 11, ASIC: 2 Channel: 11';
            if (portNumber === 12) return 'ASIC: 5 Channel: 11, ASIC: 6 Channel: 11';
            if (portNumber === 13) return 'ASIC: 3 Channel: 11, ASIC: 4 Channel: 11';
            if (portNumber === 14) return 'ASIC: 7 Channel: 11, ASIC: 8 Channel: 11';
            break;

        case 'P150_QB_GLOBAL':
        case 'P150_QB_AMERICA':
            // P150 nodes: 4 ports per tray, specific channel mapping
            if (portNumber === 1) return 'ASIC: 0 Channel: 9, ASIC: 0 Channel: 11';
            if (portNumber === 2) return 'ASIC: 0 Channel: 8, ASIC: 0 Channel: 10';
            if (portNumber === 3) return 'ASIC: 0 Channel: 5, ASIC: 0 Channel: 7';
            if (portNumber === 4) return 'ASIC: 0 Channel: 4, ASIC: 0 Channel: 6';
            break;
    }

    return `Eth${portNumber - 1}`; // Default fallback
}

function updateDeleteButtonState() {
    const deleteBtn = document.getElementById('deleteConnectionBtn');

    // Only enable the button if we're in editing mode AND have a selected connection
    if (isEdgeCreationMode && selectedConnection && selectedConnection.length > 0) {
        deleteBtn.disabled = false;
        deleteBtn.style.opacity = '1';
        deleteBtn.style.cursor = 'pointer';
    } else {
        deleteBtn.disabled = true;
        deleteBtn.style.opacity = '0.5';
        deleteBtn.style.cursor = 'not-allowed';
    }
}

// ===== Utility Functions =====

/**
 * Format rack and shelf_u numbers with zero-padding
 */
function formatRackNum(rackNum) {
    return rackNum !== undefined && rackNum !== null ? rackNum.toString().padStart(2, '0') : '';
}

function formatShelfU(shelfU) {
    return shelfU !== undefined && shelfU !== null ? shelfU.toString().padStart(2, '0') : '';
}

/**
 * Build hierarchical location label from location components
 * @param {string} hall - Hall identifier
 * @param {string} aisle - Aisle identifier
 * @param {number|string} rackNum - Rack number
 * @param {number|string} shelfU - Shelf U position (optional)
 * @returns {string} Formatted location label (e.g., "H1A205" or "H1A205U12")
 */
function buildLocationLabel(hall, aisle, rackNum, shelfU = null) {
    if (!hall || !aisle || rackNum === undefined || rackNum === null) {
        return '';
    }

    const rackPadded = formatRackNum(rackNum);
    const label = `${hall}${aisle}${rackPadded}`;

    if (shelfU !== null && shelfU !== undefined && shelfU !== '') {
        const shelfUPadded = formatShelfU(shelfU);
        return `${label}U${shelfUPadded}`;
    }

    return label;
}

/**
 * Get location data from a node or its parent hierarchy
 * @param {Object} node - Cytoscape node
 * @returns {Object} Location data {hall, aisle, rack_num, shelf_u, hostname}
 */
function getNodeLocationData(node) {
    const data = node.data();

    // If node has all location data, return it
    if (data.hall && data.aisle && data.rack_num !== undefined) {
        return {
            hall: data.hall,
            aisle: data.aisle,
            rack_num: data.rack_num,
            shelf_u: data.shelf_u,
            hostname: data.hostname || ''
        };
    }

    // Try to get from parent hierarchy (for tray/port nodes)
    if (data.type === 'port' || data.type === 'tray') {
        const parentNode = node.parent();
        if (parentNode && parentNode.length > 0) {
            if (data.type === 'port') {
                // Port: go up to tray, then shelf
                const shelfNode = parentNode.parent();
                if (shelfNode && shelfNode.length > 0) {
                    return getNodeLocationData(shelfNode);
                }
            } else {
                // Tray: go up to shelf
                return getNodeLocationData(parentNode);
            }
        }
    }

    // Return whatever we have
    return {
        hall: data.hall || '',
        aisle: data.aisle || '',
        rack_num: data.rack_num,
        shelf_u: data.shelf_u,
        hostname: data.hostname || ''
    };
}

/**
 * Get display label for a node based on priority:
 * 1. Hostname (if available)
 * 2. Location format (if all location data available)
 * 3. Type-specific label (e.g., "Shelf {U}")
 * 4. Existing label
 * 5. Node ID
 * @param {Object} nodeData - Node data object
 * @returns {string} Display label
 */
function getNodeDisplayLabel(nodeData) {
    // Priority 1: Hostname
    if (nodeData.hostname) {
        return nodeData.hostname;
    }

    // Priority 2: Location format (Hall-Aisle-Rack-ShelfU)
    if (nodeData.hall && nodeData.aisle && nodeData.rack_num !== undefined) {
        const shelfU = nodeData.shelf_u;
        if (shelfU !== undefined && shelfU !== null && shelfU !== '') {
            return buildLocationLabel(nodeData.hall, nodeData.aisle, nodeData.rack_num, shelfU);
        } else {
            return buildLocationLabel(nodeData.hall, nodeData.aisle, nodeData.rack_num);
        }
    }

    // Priority 3: Type-specific labels
    if (nodeData.type === 'shelf' && nodeData.shelf_u !== undefined) {
        return `Shelf ${nodeData.shelf_u}`;
    }

    // Priority 4: Existing label
    if (nodeData.label) {
        return nodeData.label;
    }

    // Priority 5: Fallback to ID
    return nodeData.id;
}

// ===== Event Handler Helpers =====

/**
 * Handle port click in editing mode
 */
function handlePortClickEditMode(node, evt) {
    const portId = node.id();
    const existingConnections = cy.edges(`[source="${portId}"], [target="${portId}"]`);

    // If port is already connected, select the connection
    if (existingConnections.length > 0) {
        const edge = existingConnections[0]; // Only one connection per port

        // Clear source port selection if any
        if (sourcePort) {
            sourcePort.removeClass('source-selected');
            sourcePort = null;
        }

        // Select this connection
        if (selectedConnection) {
            selectedConnection.removeClass('selected-connection');
        }
        selectedConnection = edge;
        edge.addClass('selected-connection');

        // Show port info and update UI
        showNodeInfo(node, evt.renderedPosition || evt.position);
        updateDeleteButtonState();
        return;
    }

    // Port is unconnected - handle connection creation
    if (!sourcePort) {
        // First click - select source port
        sourcePort = node;
        sourcePort.addClass('source-selected');
    } else {
        // Second click - create connection
        const targetPort = node;

        // Can't connect port to itself
        if (sourcePort.id() === targetPort.id()) {
            sourcePort.removeClass('source-selected');
            sourcePort = null;
            return;
        }

        // Create the connection
        createConnection(sourcePort.id(), targetPort.id());

        // Clear source port selection
        sourcePort.removeClass('source-selected');
        sourcePort = null;
    }
}

/**
 * Handle port click in view mode (not editing)
 */
function handlePortClickViewMode(node, evt) {
    const portId = node.id();
    const connectedEdges = cy.edges(`[source="${portId}"], [target="${portId}"]`);

    if (connectedEdges.length > 0) {
        // Port has connection - select it
        const edge = connectedEdges[0]; // Only one connection per port

        if (selectedConnection) {
            selectedConnection.removeClass('selected-connection');
        }

        selectedConnection = edge;
        edge.addClass('selected-connection');
        showNodeInfo(node, evt.renderedPosition || evt.position);
    } else {
        // Port has no connection - just show info
        if (selectedConnection) {
            selectedConnection.removeClass('selected-connection');
            selectedConnection = null;
            updateDeleteButtonState();
        }
        showNodeInfo(node, evt.renderedPosition || evt.position);
    }
}

/**
 * Clear all selection state
 */
function clearAllSelections() {
    hideNodeInfo();

    if (sourcePort) {
        sourcePort.removeClass('source-selected');
        sourcePort = null;
    }

    if (selectedConnection) {
        selectedConnection.removeClass('selected-connection');
        selectedConnection = null;
        updateDeleteButtonState();
    }
}

// ===== End Event Handler Helpers =====

function getPortLocationInfo(portNode) {
    /**
     * Build detailed location string for a port
     * @param {Object} portNode - Cytoscape port node
     * @returns {String} Formatted location string
     */
    const portLabel = portNode.data('label');
    const trayNode = portNode.parent();
    const shelfNode = trayNode.parent();

    // Get location data from the hierarchy
    const hostname = shelfNode.data('hostname') || '';
    const hall = shelfNode.data('hall') || '';
    const aisle = shelfNode.data('aisle') || '';
    const rackNum = shelfNode.data('rack_num');
    const shelfU = shelfNode.data('shelf_u');
    const trayLabel = trayNode.data('label');

    // Build location string
    let locationParts = [];

    // Prefer location format (Hall-Aisle-Rack-Shelf) as default
    if (hall && aisle && rackNum !== undefined && shelfU !== undefined) {
        // Use location format: HallAisle##U##
        locationParts.push(buildLocationLabel(hall, aisle, rackNum, shelfU));
    } else if (hostname) {
        // Fallback to hostname if location info is unavailable
        locationParts.push(hostname);
    } else if (shelfNode.data('label')) {
        // Final fallback to shelf label
        locationParts.push(shelfNode.data('label'));
    }

    // Add tray and port info
    locationParts.push(trayLabel);
    locationParts.push(portLabel);

    return locationParts.join(' › ');
}

function deleteSelectedConnection() {
    if (!selectedConnection || selectedConnection.length === 0) {
        alert('Please select a connection first by clicking on it.');
        return;
    }

    const edge = selectedConnection;
    const sourceNode = cy.getElementById(edge.data('source'));
    const targetNode = cy.getElementById(edge.data('target'));

    const sourceInfo = getPortLocationInfo(sourceNode);
    const targetInfo = getPortLocationInfo(targetNode);

    const message = `Delete connection between:\n\nSource: ${sourceInfo}\n\nTarget: ${targetInfo}`;

    if (confirm(message)) {
        edge.remove();
        selectedConnection = null;
        updateDeleteButtonState();
        updatePortConnectionStatus();
        updatePortEditingHighlight();
    }
}

function updatePortConnectionStatus() {
    // Reset all ports to default state
    cy.nodes('.port').removeClass('connected-port');

    // Mark ports that have connections
    cy.edges().forEach(function (edge) {
        const sourceId = edge.data('source');
        const targetId = edge.data('target');
        cy.getElementById(sourceId).addClass('connected-port');
        cy.getElementById(targetId).addClass('connected-port');
    });
}

function createConnection(sourceId, targetId) {
    const sourceNode = cy.getElementById(sourceId);
    const targetNode = cy.getElementById(targetId);

    if (!sourceNode.length || !targetNode.length) {
        console.error('Source or target node not found');
        return;
    }

    // Check if either port already has a connection
    const sourceConnections = cy.edges(`[source="${sourceId}"], [target="${sourceId}"]`);
    const targetConnections = cy.edges(`[source="${targetId}"], [target="${targetId}"]`);

    if (sourceConnections.length > 0) {
        alert(`Cannot create connection: Source port "${sourceNode.data('label')}" is already connected.\n\nEach port can only have one connection. Please disconnect the existing connection first.`);
        return;
    }

    if (targetConnections.length > 0) {
        alert(`Cannot create connection: Target port "${targetNode.data('label')}" is already connected.\n\nEach port can only have one connection. Please disconnect the existing connection first.`);
        return;
    }

    // Determine connection color based on whether ports are on the same node
    // Ports are 2 levels separated from nodes (node -> tray -> port)
    const sourceGrandparent = getParentAtLevel(sourceNode, 2);
    const targetGrandparent = getParentAtLevel(targetNode, 2);

    let connectionColor;
    if (sourceGrandparent && targetGrandparent && sourceGrandparent.id() === targetGrandparent.id()) {
        connectionColor = CONNECTION_COLORS.INTRA_NODE;
    } else {
        connectionColor = CONNECTION_COLORS.INTER_NODE;
    }

    const edgeId = `edge_${sourceId}_${targetId}_${Date.now()}`;
    const connectionNumber = getNextConnectionNumber();
    const newEdge = {
        data: {
            id: edgeId,
            source: sourceId,
            target: targetId,
            cable_type: DEFAULT_CABLE_CONFIG.type,
            cable_length: DEFAULT_CABLE_CONFIG.length,
            label: `#${connectionNumber}`,
            connection_number: connectionNumber,
            color: connectionColor
        }
    };

    cy.add(newEdge);

    // Update port connection status visual indicators
    updatePortConnectionStatus();

    // Update port editing highlights if in editing mode
    updatePortEditingHighlight();

    // Apply curve styling to new edge
    setTimeout(() => forceApplyCurveStyles(), 50);
}

function updateAddNodeButtonState() {
    const addNodeBtn = document.getElementById('addNodeBtn');
    const addNodeText = addNodeBtn.nextElementSibling;

    if (cy && currentData) {
        // Enable the button
        addNodeBtn.disabled = false;
        addNodeBtn.style.background = '#007bff';
        addNodeBtn.style.cursor = 'pointer';
        addNodeBtn.style.opacity = '1';
        addNodeText.textContent = 'Creates a new node with trays and ports';
    } else {
        // Disable the button
        addNodeBtn.disabled = true;
        addNodeBtn.style.background = '#6c757d';
        addNodeBtn.style.cursor = 'not-allowed';
        addNodeBtn.style.opacity = '0.6';
        addNodeText.textContent = 'Upload a visualization first to enable node creation';
    }
}

function createEmptyVisualization() {
    // Hide upload section
    const uploadSection = document.getElementById('uploadSection');
    const cyLoading = document.getElementById('cyLoading');

    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    if (cyLoading) {
        cyLoading.style.display = 'none';
    }

    // Create empty data structure that matches what initVisualization expects
    currentData = {
        nodes: [],
        edges: [],
        metadata: {
            total_connections: 0,
            total_nodes: 0
        }
    };

    // Initialize Cytoscape with empty data
    initVisualization(currentData);

    // Enable the Add Node button
    updateAddNodeButtonState();

    // Open the Cabling Editor section
    if (typeof toggleCollapsible === 'function') {
        const cablingEditorContent = document.getElementById('cablingEditor');
        if (cablingEditorContent && cablingEditorContent.classList.contains('collapsed')) {
            toggleCollapsible('cablingEditor');
        }
    }

    // Enable Connection Editing (suppress alert since we're showing custom success message)
    const toggleBtn = document.getElementById('toggleEdgeHandlesBtn');
    if (toggleBtn && toggleBtn.textContent.includes('Enable')) {
        toggleEdgeHandles(true);
    }

    // Show success message
    showSuccess('Empty visualization created! Connection editing is enabled. You can now add nodes using the "Add New Node" section.');
}

function resetLayout() {
    /**
     * Reset and recalculate layout for all nodes based on rack/shelf hierarchy
     */
    if (!cy) {
        alert('No visualization loaded. Please upload a CSV file first.');
        return;
    }

    // Get all racks and sort by rack_num
    const racks = cy.nodes('[type="rack"]');
    if (racks.length === 0) {
        // No racks - this might be 8-column format with standalone shelves
        alert('No rack hierarchy found.');
        return;
    }

    // Show status message
    showExportStatus('Resetting layout...', 'info');

    // Sort racks by rack number
    const sortedRacks = [];
    racks.forEach(function (rack) {
        sortedRacks.push({
            node: rack,
            rack_num: parseInt(rack.data('rack_num')) || 0
        });
    });
    sortedRacks.sort(function (a, b) {
        return a.rack_num - b.rack_num;
    });

    // Dynamically calculate layout constants based on actual node sizes
    // This ensures proper spacing regardless of node type (wh_galaxy, n300_lb, etc.)

    // Calculate average/max rack width
    let maxRackWidth = 0;
    racks.forEach(function (rack) {
        const bb = rack.boundingBox();
        const width = bb.w;
        if (width > maxRackWidth) maxRackWidth = width;
    });
    const rackWidth = maxRackWidth || LAYOUT_CONSTANTS.DEFAULT_RACK_WIDTH;

    // Calculate average/max shelf height (including all descendants)
    let maxShelfHeight = 0;
    let totalShelfHeight = 0;
    let shelfCount = 0;

    racks.forEach(function (rack) {
        rack.children('[type="shelf"]').forEach(function (shelf) {
            // Get bounding box of shelf including all its children
            const bb = shelf.boundingBox({ includeLabels: false, includeOverlays: false });
            const height = bb.h;
            totalShelfHeight += height;
            shelfCount++;
            if (height > maxShelfHeight) maxShelfHeight = height;
        });
    });

    const avgShelfHeight = shelfCount > 0 ? totalShelfHeight / shelfCount : 300;

    // Use max shelf height + 25% buffer for spacing to prevent any overlaps
    const shelfSpacingBuffer = 1.25; // 25% extra space
    const shelfSpacing = Math.max(maxShelfHeight, avgShelfHeight) * shelfSpacingBuffer;

    // Rack spacing should be enough for the widest rack + buffer
    const rackSpacing = rackWidth * LAYOUT_CONSTANTS.RACK_SPACING_BUFFER - rackWidth;

    // Calculate appropriate starting positions with padding
    const startX = Math.max(LAYOUT_CONSTANTS.MIN_START_X, rackWidth / 2 + LAYOUT_CONSTANTS.RACK_X_OFFSET);
    const startY = Math.max(LAYOUT_CONSTANTS.MIN_START_Y, maxShelfHeight + LAYOUT_CONSTANTS.RACK_Y_OFFSET);

    // First pass: calculate all new positions and deltas BEFORE making any changes
    const positionUpdates = [];

    sortedRacks.forEach(function (rackData, rackIndex) {
        const rack = rackData.node;

        // Calculate rack position (horizontal sequence)
        const rackX = startX + (rackIndex * (rackWidth + rackSpacing));
        const rackY = startY;

        positionUpdates.push({
            node: rack,
            newPos: { x: rackX, y: rackY }
        });

        // Get all shelves in this rack and sort by shelf_u (descending - higher U at top)
        const shelves = rack.children('[type="shelf"]');
        const sortedShelves = [];
        shelves.forEach(function (shelf) {
            sortedShelves.push({
                node: shelf,
                shelf_u: parseInt(shelf.data('shelf_u')) || 0,
                oldPos: { x: shelf.position().x, y: shelf.position().y }
            });
        });
        sortedShelves.sort(function (a, b) {
            return b.shelf_u - a.shelf_u; // Descending: higher shelf_u at top
        });

        // Calculate vertical positions for shelves (centered in rack)
        const numShelves = sortedShelves.length;
        if (numShelves > 0) {
            const totalShelfHeight = (numShelves - 1) * shelfSpacing;
            const shelfStartY = rackY - (totalShelfHeight / 2);

            // Calculate position for each shelf and its descendants
            sortedShelves.forEach(function (shelfData, shelfIndex) {
                const shelf = shelfData.node;
                const oldShelfPos = shelfData.oldPos;
                const newShelfX = rackX;
                const newShelfY = shelfStartY + (shelfIndex * shelfSpacing);

                // Calculate deltas for repositioning child nodes
                const deltaX = newShelfX - oldShelfPos.x;
                const deltaY = newShelfY - oldShelfPos.y;

                // Store shelf position update
                positionUpdates.push({
                    node: shelf,
                    newPos: { x: newShelfX, y: newShelfY }
                });

                // Store position updates for all descendants (trays and ports)
                shelf.descendants().forEach(function (child) {
                    const childPos = child.position();
                    positionUpdates.push({
                        node: child,
                        newPos: {
                            x: childPos.x + deltaX,
                            y: childPos.y + deltaY
                        }
                    });
                });
            });
        }
    });

    // Second pass: apply all position updates in a batch
    cy.startBatch();
    positionUpdates.forEach(function (update) {
        update.node.position(update.newPos);
    });
    cy.endBatch();

    // Force a complete refresh of the cytoscape instance
    cy.resize();
    cy.forceRender();

    // Small delay to ensure rendering is complete before fitting viewport and reapplying styles
    setTimeout(function () {
        // Reapply edge curve styles after repositioning
        forceApplyCurveStyles();

        // Update port connection status visual indicators
        updatePortConnectionStatus();

        // Fit the view to show all nodes with padding
        cy.fit(50);
        cy.center();

        // Force another render to ensure everything is updated
        cy.forceRender();

        // Show success message
        showExportStatus('Layout reset successfully! All nodes repositioned based on hierarchy.', 'success');
    }, 100);
}

function addNewNode() {
    const nodeTypeSelect = document.getElementById('nodeTypeSelect');
    const hostnameInput = document.getElementById('nodeHostnameInput');
    const hallInput = document.getElementById('nodeHallInput');
    const aisleInput = document.getElementById('nodeAisleInput');
    const rackInput = document.getElementById('nodeRackInput');
    const shelfUInput = document.getElementById('nodeShelfUInput');

    const nodeType = nodeTypeSelect.value;
    const hostname = hostnameInput.value.trim();
    const hall = hallInput.value.trim();
    const aisle = aisleInput.value.trim();
    const rack = parseInt(rackInput.value) || 0;
    const shelfU = parseInt(shelfUInput.value) || 0;

    // Check if cytoscape is initialized
    if (!cy) {
        alert('Please upload a CSV file and generate a visualization first before adding new nodes.');
        return;
    }

    // Validation: Either hostname OR all location fields must be filled
    const hasHostname = hostname.length > 0;
    const hasLocation = hall.length > 0 && aisle.length > 0 && rack > 0 && shelfU > 0;

    if (!hasHostname && !hasLocation) {
        alert('Please enter either a hostname OR all location fields (Hall, Aisle, Rack, Shelf U).');
        if (!hostname) hostnameInput.focus();
        return;
    }

    // Allow both hostname and location to be filled - hostname takes precedence for label

    // Check for existing node with same hostname or location
    if (hasHostname) {
        const existingNode = cy.nodes(`[hostname="${hostname}"]`);
        if (existingNode.length > 0) {
            alert(`A node with hostname "${hostname}" already exists. Please choose a different hostname.`);
            hostnameInput.focus();
            return;
        }
    } else {
        // Check for existing node with same location
        const existingNode = cy.nodes(`[hall="${hall}"][aisle="${aisle}"][rack_num="${rack}"][shelf_u="${shelfU}"]`);
        if (existingNode.length > 0) {
            alert(`A node already exists at Hall: ${hall}, Aisle: ${aisle}, Rack: ${rack}, Shelf U: ${shelfU}. Please choose a different location.`);
            return;
        }
    }

    const config = NODE_CONFIGS[nodeType];
    if (!config) {
        alert(`Unknown node type: ${nodeType}`);
        return;
    }

    // If location data is provided, ensure a rack exists for it
    let rackParentId = null;
    if (hasLocation) {
        const rackId = `rack_${rack.toString().padStart(2, '0')}`;
        let rackNode = cy.getElementById(rackId);

        // Create rack if it doesn't exist
        if (rackNode.length === 0) {
            // Calculate position for new rack
            let newRackX = 0;
            let newRackY = 0;
            const existingRacks = cy.nodes('[type="rack"]');

            if (existingRacks.length > 0) {
                // Position new rack to the right of the rightmost rack
                let maxX = -Infinity;
                existingRacks.forEach(function (rack) {
                    const rackPos = rack.position();
                    if (rackPos.x > maxX) {
                        maxX = rackPos.x;
                        newRackY = rackPos.y;
                    }
                });
                newRackX = maxX + 500; // 500px spacing between racks
            } else {
                // First rack - place at a reasonable starting position
                newRackX = LAYOUT_CONSTANTS.NEW_RACK_DEFAULT_X;
                newRackY = LAYOUT_CONSTANTS.NEW_RACK_DEFAULT_Y;
            }

            const rackLabel = `Rack ${rack}`;
            cy.add({
                group: 'nodes',
                data: {
                    id: rackId,
                    label: rackLabel,
                    type: 'rack',
                    rack_num: rack,
                    hall: hall,
                    aisle: aisle
                },
                classes: 'rack',
                position: {
                    x: newRackX,
                    y: newRackY
                }
            });
            rackNode = cy.getElementById(rackId);
        }

        rackParentId = rackId;
    }

    // Find a good position for the new node
    let newX, newY;

    if (rackParentId) {
        // Position shelf within the rack
        const rackNode = cy.getElementById(rackParentId);
        const rackPos = rackNode.position();

        // Get existing shelves in this rack to determine vertical position
        const shelvesInRack = rackNode.children('[type="shelf"]');
        const shelfCount = shelvesInRack.length;
        const shelfSpacing = 140;

        newX = rackPos.x;
        newY = rackPos.y - (shelfCount * shelfSpacing / 2); // Position based on shelf count
    } else {
        // 8-column format or no rack - position to the right of existing shelves
        const existingShelves = cy.nodes('.shelf');
        let maxX = 0;
        existingShelves.forEach(shelf => {
            const pos = shelf.position();
            if (pos.x > maxX) maxX = pos.x;
        });
        newX = maxX + 500;
        newY = 200;
    }

    // Create shelf node
    let shelfId, nodeLabel, nodeData;

    if (hasHostname) {
        // Use hostname as label, but include location data if provided
        shelfId = `${hostname}`;
        nodeLabel = hostname;
        nodeData = {
            id: shelfId,
            label: nodeLabel,
            type: 'shelf',
            hostname: hostname,
            shelf_node_type: nodeType
        };

        // Add location data if provided
        if (hasLocation) {
            nodeData.hall = hall;
            nodeData.aisle = aisle;
            nodeData.rack_num = rack;
            nodeData.shelf_u = shelfU;
        }
    } else {
        // Format: HallAisle{2-digit Rack}U{2-digit Shelf U}
        nodeLabel = buildLocationLabel(hall, aisle, rack, shelfU);
        shelfId = nodeLabel; // Use the same format as the label
        nodeData = {
            id: shelfId,
            label: nodeLabel,
            type: 'shelf',
            hall: hall,
            aisle: aisle,
            rack_num: rack,
            shelf_u: shelfU,
            shelf_node_type: nodeType
        };
    }

    // Add parent rack if it exists
    if (rackParentId) {
        nodeData.parent = rackParentId;
    }

    const shelfNode = {
        data: nodeData,
        position: { x: newX, y: newY },
        classes: 'shelf'
    };

    const nodesToAdd = [shelfNode];

    // Layout constants (based on existing cytoscape templates)
    const trayHeight = 60;
    const traySpacing = 10;
    const portWidth = 45;
    const portSpacing = 5;

    // Create trays and ports based on node configuration
    for (let trayNum = 1; trayNum <= config.tray_count; trayNum++) {
        const trayId = `${shelfId}-tray${trayNum}`;

        // Calculate tray position based on node type configuration
        let trayX, trayY;
        if (config.tray_layout === 'vertical') {
            // Vertical arrangement: T1 at top, T2, T3, T4 going down
            trayX = newX;
            trayY = newY - 150 + (trayNum - 1) * (trayHeight + traySpacing);
        } else {
            // Horizontal arrangement: T1, T2, T3, T4 arranged left-to-right
            trayX = newX - 150 + (trayNum - 1) * (trayHeight + traySpacing);
            trayY = newY;
        }

        const trayData = {
            id: trayId,
            parent: shelfId,
            label: `T${trayNum}`,
            type: 'tray',
            tray: trayNum,
            shelf_node_type: nodeType
        };

        // Add location data if available
        if (hasLocation) {
            trayData.rack_num = rack;
            trayData.shelf_u = shelfU;
            trayData.hall = hall;
            trayData.aisle = aisle;
        }
        if (hasHostname) {
            trayData.hostname = hostname;
        }

        const trayNode = {
            data: trayData,
            position: { x: trayX, y: trayY },
            classes: 'tray'
        };
        nodesToAdd.push(trayNode);

        // Create ports based on node configuration
        const portsPerTray = config.ports_per_tray;
        for (let portNum = 1; portNum <= portsPerTray; portNum++) {
            const portId = `${shelfId}-tray${trayNum}-port${portNum}`;

            // Calculate port position (orthogonal to tray arrangement)
            let portX, portY;
            if (config.tray_layout === 'vertical') {
                // Vertical trays → horizontal ports
                portX = trayX - 120 + (portNum - 1) * (portWidth + portSpacing);
                portY = trayY;
            } else {
                // Horizontal trays → vertical ports
                portX = trayX;
                portY = trayY - 100 + (portNum - 1) * (portWidth + portSpacing);
            }

            const portData = {
                id: portId,
                parent: trayId,
                label: `P${portNum}`,
                type: 'port',
                tray: trayNum,
                port: portNum,
                shelf_node_type: nodeType
            };

            // Add location data if available
            if (hasLocation) {
                portData.rack_num = rack;
                portData.shelf_u = shelfU;
                portData.hall = hall;
                portData.aisle = aisle;
            }
            if (hasHostname) {
                portData.hostname = hostname;
            }

            const portNode = {
                data: portData,
                position: { x: portX, y: portY },
                classes: 'port'
            };
            nodesToAdd.push(portNode);
        }
    }

    try {
        // Add all nodes to cytoscape
        cy.add(nodesToAdd);

        // Make trays and ports non-draggable
        cy.nodes('.tray, .port').ungrabify();

        // Apply styling and layout
        setTimeout(() => {
            forceApplyCurveStyles();
            updatePortConnectionStatus();
            updatePortEditingHighlight(); // Highlight available ports if in editing mode
        }, 100);

        // Clear all inputs
        hostnameInput.value = '';
        hallInput.value = '';
        aisleInput.value = '';
        rackInput.value = '';
        shelfUInput.value = '';

        // Show success message
        const nodeDescription = hasHostname ? `"${hostname}"` : `"${nodeLabel}"`;
        const locationInfo = hasHostname && hasLocation ? ` (with location: ${buildLocationLabel(hall, aisle, rack, shelfU)})` : '';
        alert(`Successfully added ${nodeType} node ${nodeDescription}${locationInfo} with ${config.tray_count} trays.`);

    } catch (error) {
        console.error('Error adding new node:', error);
        alert(`Failed to add node: ${error.message}`);
    }
}

function toggleEdgeHandles() {
    const btn = document.getElementById('toggleEdgeHandlesBtn');

    if (!cy) {
        console.error('Cytoscape instance not available');
        return;
    }

    if (btn.textContent.includes('Enable')) {
        // Enable connection creation mode
        isEdgeCreationMode = true;
        btn.textContent = '🔗 Disable Connection Editing';
        btn.style.backgroundColor = '#dc3545';

        // Show delete connection section
        document.getElementById('deleteConnectionSection').style.display = 'block';

        // Show add node section
        document.getElementById('addNodeSection').style.display = 'block';

        // Add visual feedback only for available (unconnected) ports
        updatePortEditingHighlight();

        // Show instruction
        alert('Connection editing enabled!\n\n• Click unconnected port → Click another port = Create connection\n• Click connection to select it, then use Delete button or Backspace/Delete key\n• Click empty space = Cancel selection\n\nNote: Only unconnected ports are highlighted in orange');

    } else {
        // Disable connection creation mode
        isEdgeCreationMode = false;

        // Clear source port selection and remove styling
        if (sourcePort) {
            sourcePort.removeClass('source-selected');
        }
        sourcePort = null;

        btn.textContent = '🔗 Enable Connection Editing';
        btn.style.backgroundColor = '#28a745';

        // Hide delete connection section
        document.getElementById('deleteConnectionSection').style.display = 'none';

        // Hide add node section
        document.getElementById('addNodeSection').style.display = 'none';

        // Clear any selected connection and remove its styling
        if (selectedConnection) {
            selectedConnection.removeClass('selected-connection');
        }
        selectedConnection = null;
        document.getElementById('deleteConnectionBtn').disabled = true;
        document.getElementById('deleteConnectionBtn').style.opacity = '0.5';

        // Remove visual feedback from all ports
        cy.nodes('.port').style({
            'border-width': '2px',
            'border-color': '#666666',
            'border-opacity': 1.0
        });

        // Remove any source port highlighting (redundant but safe)
        cy.nodes('.port').removeClass('source-selected');

        // Remove selected-connection class from all edges
        cy.edges().removeClass('selected-connection');
    }
}

function updatePortEditingHighlight() {
    if (!isEdgeCreationMode) return;

    cy.nodes('.port').forEach(port => {
        const portId = port.id();
        const connections = cy.edges(`[source="${portId}"], [target="${portId}"]`);

        if (connections.length === 0) {
            // Port is available - add orange highlighting
            port.style({
                'border-width': '3px',
                'border-color': '#ff6600',
                'border-opacity': 0.7
            });
        } else {
            // Port is connected - use default styling
            port.style({
                'border-width': '2px',
                'border-color': '#666666',
                'border-opacity': 1.0
            });
        }
    });
}

// File upload handlers
const uploadSection = document.getElementById('uploadSection');
const csvFile = document.getElementById('csvFile');
const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const success = document.getElementById('success');

// Drag and drop handlers
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.endsWith('.csv')) {
        csvFile.files = files;
    }
});

csvFile.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        // File selected, button text remains "Generate Visualization"
    }
});

async function uploadFile() {
    const file = csvFile.files[0];

    if (!file) {
        showError('Please select a CSV file first.');
        return;
    }

    if (!file.name.endsWith('.csv')) {
        showError('Please select a CSV file (must end with .csv).');
        return;
    }

    // Reset any global state
    currentData = null;
    selectedConnection = null;
    isEdgeCreationMode = false;

    // Show loading state
    loading.style.display = 'block';
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processing...';
    hideMessages();

    const formData = new FormData();
    formData.append('csv_file', file);

    try {
        const response = await fetch('/upload_csv', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            currentData = result.data;

            // Check for unknown node types and show warning
            if (result.unknown_types && result.unknown_types.length > 0) {
                const unknownTypesStr = result.unknown_types.map(t => t.toUpperCase()).join(', ');
                showWarning(`Successfully processed ${file.name}!<br><strong>⚠️ Warning:</strong> Unknown node types detected and auto-configured: ${unknownTypesStr}`);
            } else {
                showSuccess(`Successfully processed ${file.name}!`);
            }

            initVisualization(result.data);

            // Enable the Add Node button after successful upload
            updateAddNodeButtonState();
        } else {
            showError(`Error: ${result.error || 'Unknown error occurred'}`);
        }
    } catch (err) {
        showError(`Upload failed: ${err.message}`);
        console.error('Upload error:', err);
    } finally {
        // Reset UI state
        loading.style.display = 'none';
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Generate Visualization';
    }
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    success.style.display = 'none';
}

function showSuccess(message) {
    success.innerHTML = message;
    success.style.display = 'block';
    success.style.backgroundColor = '#d4edda';
    success.style.borderColor = '#c3e6cb';
    success.style.color = '#155724';
    error.style.display = 'none';
}

function showWarning(message) {
    success.innerHTML = message;
    success.style.display = 'block';
    success.style.backgroundColor = '#fff3cd';
    success.style.borderColor = '#ffeaa7';
    success.style.color = '#856404';
    error.style.display = 'none';
}

function hideMessages() {
    error.style.display = 'none';
    success.style.display = 'none';
    // Reset success styling to default
    success.style.backgroundColor = '';
    success.style.borderColor = '';
    success.style.color = '';
}

function initVisualization(data) {
    // Safety check for DOM elements
    const cyLoading = document.getElementById('cyLoading');
    const cyContainer = document.getElementById('cy');

    if (!cyLoading || !cyContainer) {
        console.error('Required DOM elements not found');
        return;
    }

    cyLoading.style.display = 'none';

    console.log('Initializing Cytoscape with data:', data);
    console.log('Container element:', cyContainer);
    console.log('Container dimensions:', cyContainer.offsetWidth, 'x', cyContainer.offsetHeight);
    console.log('Elements count:', data.elements ? data.elements.length : 'undefined');

    // Ensure container has proper dimensions
    if (cyContainer.offsetWidth === 0 || cyContainer.offsetHeight === 0) {
        console.warn('Container has zero dimensions, setting explicit size');
        cyContainer.style.width = '100%';
        cyContainer.style.height = '600px';
    }

    try {
        if (typeof cy !== 'undefined' && cy) {
            // Clear existing elements and add new ones
            console.log('Clearing existing elements and adding new ones');
            cy.elements().remove();
            cy.add(data.elements);
            // Make trays and ports non-draggable
            cy.nodes('.tray, .port').ungrabify();
            cy.layout({ name: 'preset' }).run();
        } else {
            // Create new Cytoscape instance
            console.log('Creating new Cytoscape instance');
            cy = cytoscape({
                container: cyContainer,
                elements: data.elements,
                style: getCytoscapeStyles(),
                layout: { name: 'preset' },
                minZoom: 0.1,
                maxZoom: 5,
                wheelSensitivity: 0.2
            });

            // Make trays and ports non-draggable
            cy.nodes('.tray, .port').ungrabify();

            // Add event handlers for new instance
            addCytoscapeEventHandlers();
        }

        console.log('Cytoscape instance ready:', cy);
        console.log('Nodes count:', cy.nodes().length);
        console.log('Edges count:', cy.edges().length);

        // Apply final curve styling
        setTimeout(() => {
            forceApplyCurveStyles();
            updatePortConnectionStatus();
        }, 100);

        // Initialize delete button state
        updateDeleteButtonState();

        // Add remaining event handlers (only for new instances)
        if (typeof cy !== 'undefined' && cy && !window.cytoscapeEventHandlersAdded) {
            addConnectionTypeEventHandlers();
            window.cytoscapeEventHandlersAdded = true;
        }

        // Populate node filter dropdown
        populateNodeFilterDropdown();

        console.log('Cytoscape initialization complete');
    } catch (error) {
        console.error('Error initializing Cytoscape:', error);
    }
}

function forceApplyCurveStyles() {
    if (typeof cy === 'undefined' || !cy) return;

    const edges = cy.edges();
    const viewport = cy.extent();
    const viewportWidth = viewport.w;
    const viewportHeight = viewport.h;
    const baseDistance = Math.min(viewportWidth, viewportHeight) * 0.05; // 5% of smaller viewport dimension

    cy.startBatch();

    edges.forEach(function (edge) {
        const sourceNode = edge.source();
        const targetNode = edge.target();
        const sourcePos = sourceNode.position();
        const targetPos = targetNode.position();

        const sourceId = sourceNode.id();
        const targetId = targetNode.id();
        const isSameShelf = checkSameShelf(sourceId, targetId);

        if (isSameShelf) {
            // Same shelf - use bezier curves with viewport-based distance
            const dx = targetPos.x - sourcePos.x;
            const dy = targetPos.y - sourcePos.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            // Scale curve distance based on connection length and viewport
            let curveMultiplier;
            curveMultiplier = (Math.sqrt(distance) * 0.05);

            const curveDistance = `${Math.round(baseDistance * curveMultiplier)}px`;

            edge.style({
                'curve-style': 'unbundled-bezier',
                'control-point-distances': [curveDistance],
                'control-point-weights': [0.5]
            });
        } else {
            // Different shelf - use straight edges
            edge.style({
                'curve-style': 'straight'
            });
        }
    });

    cy.endBatch();

    // Force render to ensure z-index changes take effect
    cy.forceRender();
}

function checkSameShelf(sourceId, targetId) {
    if (typeof cy === 'undefined' || !cy) return false;

    const sourceNode = cy.getElementById(sourceId);
    const targetNode = cy.getElementById(targetId);

    if (!sourceNode.length || !targetNode.length) {
        return false;
    }

    // Get parent 2 levels up (port -> tray -> shelf)
    const sourceShelf = getParentAtLevel(sourceNode, 2);
    const targetShelf = getParentAtLevel(targetNode, 2);

    return sourceShelf && targetShelf && sourceShelf.id() === targetShelf.id();
}

function getParentAtLevel(node, level) {
    let currentNode = node;
    for (let i = 0; i < level; i++) {
        const parent = currentNode.parent();
        if (!parent.length) return null;
        currentNode = parent;
    }
    return currentNode;
}

function getCytoscapeStyles() {
    return [
        // Basic edge styles - high z-index to ensure above all nodes
        {
            selector: 'edge',
            style: {
                'width': 3,
                'line-color': 'data(color)',
                'line-opacity': 1,
                'curve-style': 'unbundled-bezier',
                'control-point-distances': ['100px'],
                'control-point-weights': [0.5],
                'label': 'data(label)',
                'font-size': 12,
                'font-weight': 'bold',
                'text-background-color': '#ffffff',
                'text-background-opacity': 1.0,
                'text-background-padding': 2,
                'text-border-width': 2,
                'text-border-color': '#333333',
                'text-border-opacity': 1.0,
                'z-index': 1000,
                'z-compound-depth': 'top'
            }
        },

        // Selected edge styles - highest z-index
        {
            selector: 'edge:selected',
            style: {
                'width': 4,
                'line-color': 'data(color)',
                'line-opacity': 1,
                'z-index': 2000,
                'z-compound-depth': 'top',
                'text-background-color': '#ffff99',
                'text-background-opacity': 1.0,
                'text-background-padding': 2,
                'text-border-width': 2,
                'text-border-color': '#ff6600',
                'text-border-opacity': 1.0
            }
        },

        // Style for new connections being created
        {
            selector: '.new-connection',
            style: {
                'line-color': '#ff6600',
                'width': 4,
                'line-style': 'dashed',
                'opacity': 0.8,
                'z-index': 200
            }
        },

        // Style for selected connection (for deletion)
        {
            selector: '.selected-connection',
            style: {
                'line-color': '#ff0000',
                'width': 5,
                'line-opacity': 1,
                'z-index': 190,
                'overlay-color': '#ff0000',
                'overlay-opacity': 0.3,
                'overlay-padding': 3
            }
        },

        // Style for source port selection during connection creation
        {
            selector: '.port.source-selected',
            style: {
                'background-color': '#00ff00',
                'border-color': '#00aa00',
                'border-width': '4px'
            }
        },

        // Style for ports that already have connections
        {
            selector: '.port.connected-port',
            style: {
                'background-color': '#ffcccc',
                'border-color': '#cc0000',
                'border-width': '2px'
            }
        },

        // Rack styles - large containers with dark theme (draggable)
        {
            selector: '.rack',
            style: {
                'shape': 'round-rectangle',
                'background-color': '#e6f3ff',
                'background-opacity': 0.4,
                'border-width': 4,
                'border-color': '#003366',
                'border-opacity': 1.0,
                'label': 'data(label)',
                'text-valign': 'top',
                'text-halign': 'center',
                'font-size': 20,
                'font-weight': 'bold',
                'color': '#000000',
                'text-background-color': '#ffffff',
                'text-background-opacity': 0.9,
                'text-border-width': 1,
                'text-border-color': '#333333',
                'padding': 15,
                'min-width': '380px',
                'min-height': '500px',
                'z-index': 1
            }
        },

        // Shelf unit styles - medium containers with blue theme (draggable)
        {
            selector: '.shelf',
            style: {
                'shape': 'round-rectangle',
                'background-color': '#cce7ff',
                'background-opacity': 0.6,
                'border-width': 3,
                'border-color': '#0066cc',
                'border-opacity': 1.0,
                'label': 'data(label)',
                'text-valign': 'top',
                'text-halign': 'left',
                'text-margin-x': 10,  // Left padding for label
                'text-margin-y': 8,   // Top padding for label
                'font-size': 16,
                'font-weight': 'bold',
                'color': '#003366',
                'text-background-color': '#ffffff',
                'text-background-opacity': 0.9,
                'text-background-padding': 4,  // Padding around text background for legibility
                'text-border-width': 1,
                'text-border-color': '#0066cc',
                'padding': 12,
                'z-index': 1
                // Removed fixed min-width and min-height to allow auto-sizing
            }
        },

        // Tray styles - small containers with gray theme (non-draggable)
        {
            selector: '.tray',
            style: {
                'shape': 'round-rectangle',
                'background-color': '#f0f0f0',
                'background-opacity': 0.8,
                'border-width': 2,
                'border-color': '#666666',
                'border-opacity': 1.0,
                'label': 'data(label)',
                'text-valign': 'top',
                'text-halign': 'left',
                'text-margin-x': 6,   // Left padding for label
                'text-margin-y': 6,   // Top padding for label
                'font-size': 14,
                'font-weight': 'bold',
                'color': '#333333',
                'text-background-color': '#ffffff',
                'text-background-opacity': 0.9,
                'text-background-padding': 3,  // Padding around text background for legibility
                'text-border-width': 1,
                'text-border-color': '#666666',
                'padding': 8,
                'z-index': 1
                // Removed fixed min-width and min-height to allow auto-sizing
            }
        },

        // Port styles - leaf nodes with distinct rectangular appearance (non-draggable)
        {
            selector: '.port',
            style: {
                'shape': 'rectangle',
                'background-color': '#ffffff',
                'border-width': 2,
                'border-color': '#000000',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': 12,
                'font-weight': 'bold',
                'color': '#000000',
                'width': '35px',
                'height': '25px',
                'z-index': 1
            }
        },

        // Selected state - visual highlighting only
        {
            selector: ':selected',
            style: {
                'overlay-color': '#ff0000',
                'overlay-opacity': 0.3,
                'overlay-padding': 5
            }
        }
    ];
}

function addCytoscapeEventHandlers() {
    // Node click handler for info display and port connection creation
    cy.on('tap', 'node', function (evt) {
        const node = evt.target;

        // Handle port clicks
        if (node.hasClass('port')) {
            if (isEdgeCreationMode) {
                handlePortClickEditMode(node, evt);
            } else {
                handlePortClickViewMode(node, evt);
            }
        } else {
            // Non-port node clicked - clear connection selection
            if (selectedConnection) {
                selectedConnection.removeClass('selected-connection');
                selectedConnection = null;
                updateDeleteButtonState();
            }
            showNodeInfo(node, evt.renderedPosition || evt.position);
        }
    });

    // Double-click handler for inline editing of shelf node details
    cy.on('dbltap', 'node', function (evt) {
        const node = evt.target;
        const data = node.data();

        // Only allow editing for shelf nodes
        if (data.type === 'shelf') {
            enableShelfEditing(node, evt.renderedPosition || evt.position);
        }
    });

    // Edge click handler for connection selection and info display
    cy.on('tap', 'edge', function (evt) {
        const edge = evt.target;
        const position = evt.renderedPosition || evt.position;

        // Clear source port selection if clicking on an edge (editing mode only)
        if (isEdgeCreationMode && sourcePort) {
            sourcePort.removeClass('source-selected');
            sourcePort = null;
        }

        // Deselect previously selected connection (works in both editing and normal mode)
        if (selectedConnection) {
            selectedConnection.removeClass('selected-connection');
        }

        // Select this connection (works in both editing and normal mode)
        selectedConnection = edge;
        edge.addClass('selected-connection');

        // Update delete button state (only relevant in editing mode)
        if (isEdgeCreationMode) {
            updateDeleteButtonState();
        }

        // Show connection info annotation for any edge click (editing mode or not)
        showConnectionInfo(edge, position);
    });

    // Click on background to hide info, deselect connection, and clear source port
    cy.on('tap', function (evt) {
        if (evt.target === cy) {
            clearAllSelections();
        }
    });
}

function addConnectionTypeEventHandlers() {
    // Add event listeners to connection type checkboxes
    const checkboxes = [
        'showIntraNodeConnections',
        'showIntraRackConnections',
        'showInterRackConnections'
    ];

    checkboxes.forEach(function (checkboxId) {
        const checkbox = document.getElementById(checkboxId);
        if (checkbox) {
            checkbox.addEventListener('change', function () {
                // Reapply the current filter when checkbox changes
                const rangeInput = document.getElementById('connectionRangeString');
                if (rangeInput.value.trim() !== '') {
                    applyConnectionRange();
                } else {
                    // If no range filter is active, just apply type filter to all connections
                    applyConnectionTypeFilter();
                }
            });
        }
    });

    // Add event listener to node filter dropdown
    const nodeFilterSelect = document.getElementById('nodeFilterSelect');
    if (nodeFilterSelect) {
        nodeFilterSelect.addEventListener('change', function () {
            // Reapply the current filter when node selection changes
            const rangeInput = document.getElementById('connectionRangeString');
            if (rangeInput.value.trim() !== '') {
                applyConnectionRange();
            } else {
                // If no range filter is active, just apply node filter to all connections
                applyNodeFilter();
            }
        });
    }
}

function applyConnectionTypeFilter() {
    if (typeof cy === 'undefined' || !cy) {
        return;
    }

    // Get connection type filter settings
    const showIntraNode = document.getElementById('showIntraNodeConnections').checked;
    const showIntraRack = document.getElementById('showIntraRackConnections').checked;
    const showInterRack = document.getElementById('showInterRackConnections').checked;

    // Get all edges
    const allEdges = cy.edges();
    let visibleCount = 0;
    let hiddenCount = 0;

    // Filter edges based on connection type only
    allEdges.forEach(function (edge) {
        const sourceNode = edge.source();
        const targetNode = edge.target();

        // Get the parent shelf/node for both endpoints
        const sourceParent = getParentShelfNode(sourceNode);
        const targetParent = getParentShelfNode(targetNode);

        // Determine connection types using hierarchy traversal
        const isIntraNode = sourceParent && targetParent && sourceParent.id() === targetParent.id();

        // For rack-level filtering, get rack numbers from parent shelf nodes
        const sourceRack = sourceParent ? (sourceParent.data('rack_num') || sourceParent.data('rack') || 0) : 0;
        const targetRack = targetParent ? (targetParent.data('rack_num') || targetParent.data('rack') || 0) : 0;
        const isIntraRack = sourceRack === targetRack && sourceRack > 0 && !isIntraNode; // Intra-rack but not intra-node
        const isInterRack = sourceRack !== targetRack && sourceRack > 0 && targetRack > 0;

        // Check if connection should be visible based on type
        let shouldShowByType = false;
        if (isIntraNode && showIntraNode) shouldShowByType = true;
        if (isIntraRack && showIntraRack) shouldShowByType = true;
        if (isInterRack && showInterRack) shouldShowByType = true;

        if (shouldShowByType) {
            edge.style('display', 'element');
            visibleCount++;
        } else {
            edge.style('display', 'none');
            hiddenCount++;
        }
    });

    // Update status
    const statusDiv = document.getElementById('rangeStatus');
    statusDiv.textContent = `Showing ${visibleCount} connections (${hiddenCount} hidden by type filter)`;
    statusDiv.style.color = '#28a745';
}

function getParentShelfNode(node) {
    // Go up 2 levels: port -> tray -> shelf
    let currentNode = node;
    for (let i = 0; i < 2; i++) {
        const parent = currentNode.parent();
        if (!parent || !parent.length) {
            return null;
        }
        currentNode = parent;
    }
    return currentNode;
}

function applyNodeFilter() {
    if (typeof cy === 'undefined' || !cy) {
        return;
    }

    // Get selected node
    const nodeFilterSelect = document.getElementById('nodeFilterSelect');
    const selectedNodeId = nodeFilterSelect.value;

    // Get all edges
    const allEdges = cy.edges();
    let visibleCount = 0;
    let hiddenCount = 0;

    if (selectedNodeId === '') {
        // Show all connections
        allEdges.forEach(function (edge) {
            edge.style('display', 'element');
            visibleCount++;
        });
    } else {
        // Filter connections involving the selected node
        allEdges.forEach(function (edge) {
            const sourceNode = edge.source();
            const targetNode = edge.target();

            // Get the parent shelf/node for both endpoints (go up 2 levels: port -> tray -> shelf)
            const sourceParent = getParentShelfNode(sourceNode);
            const targetParent = getParentShelfNode(targetNode);

            // Check if either parent matches the selected node
            if (sourceParent && sourceParent.id() === selectedNodeId ||
                targetParent && targetParent.id() === selectedNodeId) {
                edge.style('display', 'element');
                visibleCount++;
            } else {
                edge.style('display', 'none');
                hiddenCount++;
            }
        });
    }

    // Update status
    const statusDiv = document.getElementById('rangeStatus');
    if (selectedNodeId === '') {
        statusDiv.textContent = `Showing all connections (${visibleCount} total)`;
        statusDiv.style.color = '#666';
    } else {
        const selectedNode = cy.getElementById(selectedNodeId);
        const nodeLabel = selectedNode.data('label') || selectedNodeId;
        statusDiv.textContent = `Showing connections for "${nodeLabel}" (${visibleCount} visible, ${hiddenCount} hidden)`;
        statusDiv.style.color = '#28a745';
    }
}

function populateNodeFilterDropdown() {
    if (typeof cy === 'undefined' || !cy) {
        return;
    }

    const nodeFilterSelect = document.getElementById('nodeFilterSelect');
    if (!nodeFilterSelect) {
        return;
    }

    // Clear existing options
    nodeFilterSelect.innerHTML = '<option value="">Show all nodes</option>';

    // Get all shelf nodes (main nodes, not trays or ports)
    const shelfNodes = cy.nodes().filter(function (node) {
        const nodeType = node.data('type');
        return nodeType === 'shelf' || nodeType === 'node';
    });

    // Add each shelf node to the dropdown
    shelfNodes.forEach(function (node) {
        const nodeId = node.id();
        const nodeData = node.data();
        let nodeLabel;

        nodeLabel = getNodeDisplayLabel(nodeData);

        const option = document.createElement('option');
        option.value = nodeId;
        option.textContent = nodeLabel;
        nodeFilterSelect.appendChild(option);
    });
}

function showNodeInfo(node, position) {
    const data = node.data();
    const nodeInfo = document.getElementById('nodeInfo');
    const content = document.getElementById('nodeInfoContent');

    let html = `<strong>${data.label || data.id}</strong><br>`;
    html += `Type: ${data.type || 'Unknown'}<br>`;

    // Show location information for rack nodes
    if (data.type === 'rack') {
        if (data.hall || data.aisle || data.rack_num) {
            html += `<br><strong>Location:</strong><br>`;
            if (data.hall !== undefined && data.hall !== '') html += `Hall: ${data.hall}<br>`;
            if (data.aisle !== undefined && data.aisle !== '') html += `Aisle: ${data.aisle}<br>`;
            if (data.rack_num !== undefined) html += `Rack: ${data.rack_num}<br>`;
        }
    }
    // Show location information based on available data
    else if (data.type === 'shelf' || data.type === 'node') {
        // For shelf/node types, show location info
        if (data.rack_num !== undefined && data.shelf_u !== undefined) {
            // 20-column format: show Hostname, Hall, Aisle, Rack, Shelf_U info
            html += `<br><strong>Location:</strong><br>`;
            if (data.hostname !== undefined && data.hostname !== '') html += `Hostname: ${data.hostname}<br>`;
            if (data.hall !== undefined && data.hall !== '') html += `Hall: ${data.hall}<br>`;
            if (data.aisle !== undefined && data.aisle !== '') html += `Aisle: ${data.aisle}<br>`;
            html += `Rack: ${data.rack_num}<br>`;
            html += `Shelf U: ${data.shelf_u}<br>`;
        } else if (data.hostname !== undefined) {
            // 8-column format: show hostname info
            html += `<br><strong>Location:</strong><br>`;
            html += `Hostname: ${data.hostname}<br>`;
        }

        // Show node type if available
        if (data.shelf_node_type) {
            html += `Node Type: ${data.shelf_node_type.toUpperCase()}<br>`;
        }
    } else {
        // For other node types (tray, port), show hierarchical location with individual fields
        const locationData = getNodeLocationData(node);

        html += `<br><strong>Location:</strong><br>`;
        if (locationData.hostname) html += `Hostname: ${locationData.hostname}<br>`;
        if (locationData.hall) html += `Hall: ${locationData.hall}<br>`;
        if (locationData.aisle) html += `Aisle: ${locationData.aisle}<br>`;
        if (locationData.rack_num !== undefined) html += `Rack: ${locationData.rack_num}<br>`;
        if (locationData.shelf_u !== undefined) html += `Shelf U: ${locationData.shelf_u}<br>`;
        if (data.tray !== undefined) html += `Tray: ${data.tray}<br>`;
        if (data.port !== undefined) html += `Port: ${data.port}<br>`;

        // Add Eth_Channel Mapping for ports
        if (data.type === 'port') {
            let nodeType = data.shelf_node_type;
            if (!nodeType) {
                const shelfNode = node.parent().parent();
                if (shelfNode && shelfNode.length > 0) {
                    nodeType = shelfNode.data('shelf_node_type');
                }
            }

            const ethChannel = getEthChannelMapping(nodeType, data.port);
            html += `Eth_Channel Mapping: ${ethChannel}<br>`;
        }
    }

    content.innerHTML = html;

    nodeInfo.style.left = `${position.x + 10}px`;
    nodeInfo.style.top = `${position.y + 10}px`;
    nodeInfo.style.display = 'block';
}

function hideNodeInfo() {
    document.getElementById('nodeInfo').style.display = 'none';
}

function showConnectionInfo(edge, position) {
    /**
     * Show detailed information about a connection
     * @param {Object} edge - Cytoscape edge element
     * @param {Object} position - Position to display the info panel
     */
    const nodeInfo = document.getElementById('nodeInfo');
    const content = document.getElementById('nodeInfoContent');

    const edgeData = edge.data();
    const sourceNode = cy.getElementById(edgeData.source);
    const targetNode = cy.getElementById(edgeData.target);

    // Get detailed location info for both endpoints
    const sourceInfo = getPortLocationInfo(sourceNode);
    const targetInfo = getPortLocationInfo(targetNode);

    // Build HTML content
    let html = `<strong>Connection Details</strong><br><br>`;

    html += `<strong>Source:</strong><br>`;
    html += `${sourceInfo}<br><br>`;

    html += `<strong>Target:</strong><br>`;
    html += `${targetInfo}<br><br>`;

    // Show cable information
    html += `<strong>Cable Info:</strong><br>`;
    html += `Type: ${edgeData.cable_type || 'Unknown'}<br>`;
    html += `Length: ${edgeData.cable_length || 'Unknown'}<br>`;

    // Show connection number if available
    if (edgeData.connection_number !== undefined) {
        html += `Connection #: ${edgeData.connection_number}<br>`;
    }

    content.innerHTML = html;

    nodeInfo.style.left = `${position.x + 10}px`;
    nodeInfo.style.top = `${position.y + 10}px`;
    nodeInfo.style.display = 'block';
}


function enableShelfEditing(node, position) {
    const data = node.data();
    const nodeInfo = document.getElementById('nodeInfo');
    const content = document.getElementById('nodeInfoContent');

    // Create editing interface for all shelf fields
    let html = `<strong>Edit Shelf Node</strong><br>`;
    html += `Node: ${data.label || data.id}<br><br>`;

    html += `<div style="margin-bottom: 10px;">`;
    html += `<label style="display: block; margin-bottom: 3px; font-weight: bold;">Hostname:</label>`;
    html += `<input type="text" id="hostnameEditInput" value="${data.hostname || ''}" placeholder="Enter hostname" style="width: 200px; padding: 5px;">`;
    html += `</div>`;

    html += `<div style="margin-bottom: 10px;">`;
    html += `<label style="display: block; margin-bottom: 3px; font-weight: bold;">Hall:</label>`;
    html += `<input type="text" id="hallEditInput" value="${data.hall || ''}" placeholder="Enter hall" style="width: 200px; padding: 5px;">`;
    html += `</div>`;

    html += `<div style="margin-bottom: 10px;">`;
    html += `<label style="display: block; margin-bottom: 3px; font-weight: bold;">Aisle:</label>`;
    html += `<input type="text" id="aisleEditInput" value="${data.aisle || ''}" placeholder="Enter aisle" style="width: 200px; padding: 5px;">`;
    html += `</div>`;

    html += `<div style="margin-bottom: 10px;">`;
    html += `<label style="display: block; margin-bottom: 3px; font-weight: bold;">Rack:</label>`;
    html += `<input type="number" id="rackEditInput" value="${data.rack_num || ''}" placeholder="Enter rack number" style="width: 200px; padding: 5px;">`;
    html += `</div>`;

    html += `<div style="margin-bottom: 10px;">`;
    html += `<label style="display: block; margin-bottom: 3px; font-weight: bold;">Shelf U:</label>`;
    html += `<input type="number" id="shelfUEditInput" value="${data.shelf_u || ''}" placeholder="Enter shelf U" style="width: 200px; padding: 5px;">`;
    html += `</div>`;

    html += `<br>`;
    html += `<button onclick="saveShelfEdit('${node.id()}')" style="background: #007bff; color: white; border: none; padding: 8px 15px; margin-right: 5px; cursor: pointer;">Save</button>`;
    html += `<button onclick="cancelShelfEdit()" style="background: #6c757d; color: white; border: none; padding: 8px 15px; cursor: pointer;">Cancel</button>`;

    content.innerHTML = html;

    // Position editing form to the right of the node
    nodeInfo.style.left = `${position.x + 20}px`;
    nodeInfo.style.top = `${position.y - 10}px`;
    nodeInfo.style.display = 'block';

    // Focus on the hostname input field
    setTimeout(() => {
        const input = document.getElementById('hostnameEditInput');
        if (input) {
            input.focus();
            input.select();
        }
    }, 100);
}

// Make functions globally accessible for onclick handlers
window.saveShelfEdit = function (nodeId) {
    const hostnameInput = document.getElementById('hostnameEditInput');
    const hallInput = document.getElementById('hallEditInput');
    const aisleInput = document.getElementById('aisleEditInput');
    const rackInput = document.getElementById('rackEditInput');
    const shelfUInput = document.getElementById('shelfUEditInput');

    const newHostname = hostnameInput.value.trim();
    const newHall = hallInput.value.trim();
    const newAisle = aisleInput.value.trim();
    const newRack = rackInput.value ? parseInt(rackInput.value) : undefined;
    const newShelfU = shelfUInput.value ? parseInt(shelfUInput.value) : undefined;

    // Get the node to check for changes
    const node = cy.getElementById(nodeId);
    const oldHostname = node.data('hostname') || '';
    const oldHall = node.data('hall') || '';
    const oldAisle = node.data('aisle') || '';
    const oldRack = node.data('rack_num');
    const oldShelfU = node.data('shelf_u');

    // Check if at least one field has changed
    const hostnameChanged = newHostname !== oldHostname;
    const hallChanged = newHall !== oldHall;
    const aisleChanged = newAisle !== oldAisle;
    const rackChanged = newRack !== oldRack && newRack !== undefined;
    const shelfUChanged = newShelfU !== oldShelfU && newShelfU !== undefined;

    if (!hostnameChanged && !hallChanged && !aisleChanged && !rackChanged && !shelfUChanged) {
        alert('No changes detected. Please modify at least one field.');
        return;
    }

    // Check if hostname already exists on another node (if hostname changed)
    if (hostnameChanged && newHostname) {
        let hostnameExists = false;
        cy.nodes().forEach(function (n) {
            if (n.id() !== nodeId && n.data('hostname') === newHostname) {
                hostnameExists = true;
            }
        });

        if (hostnameExists) {
            alert(`Hostname "${newHostname}" already exists on another node. Please choose a different hostname.`);
            return;
        }
    }

    // Update the node data only with changed values
    if (hostnameChanged) {
        node.data('hostname', newHostname);
        // Update hostname for all child nodes (trays and ports)
        node.descendants().forEach(function (child) {
            child.data('hostname', newHostname);
        });
    }
    if (hallChanged) node.data('hall', newHall);
    if (aisleChanged) node.data('aisle', newAisle);
    if (rackChanged) node.data('rack_num', newRack);
    if (shelfUChanged) {
        node.data('shelf_u', newShelfU);
        // Update shelf_u for all child nodes (trays and ports)
        node.descendants().forEach(function (child) {
            child.data('shelf_u', newShelfU);
        });
    }

    // Update the node label - use the current state of the node after updates
    let newLabel;
    const currentHostname = node.data('hostname') || '';
    const currentHall = node.data('hall') || '';
    const currentAisle = node.data('aisle') || '';
    const currentRack = node.data('rack_num');
    const currentShelfU = node.data('shelf_u');

    // Prefer hostname for display label, fall back to location format
    if (currentHostname) {
        newLabel = currentHostname;
    } else if (currentHall && currentAisle && currentRack !== undefined && currentShelfU !== undefined) {
        newLabel = buildLocationLabel(currentHall, currentAisle, currentRack, currentShelfU);
    } else if (currentShelfU !== undefined) {
        newLabel = `Shelf ${currentShelfU}`;
    } else {
        newLabel = node.data('label'); // Keep existing label
    }
    node.data('label', newLabel);

    // Handle rack change - move shelf to new rack if rack number changed
    const parent = node.parent();
    if (rackChanged && currentRack !== undefined) {
        // Need to move this shelf to a different rack
        const newRackId = `rack_${currentRack.toString().padStart(2, '0')}`;
        let newRackNode = cy.getElementById(newRackId);

        // If the new rack doesn't exist, create it
        if (newRackNode.length === 0) {
            const rackLabel = `Rack ${currentRack}`;

            // Calculate position for new rack - place it relative to existing racks
            let newRackX = 0;
            let newRackY = 0;
            const existingRacks = cy.nodes('[type="rack"]');

            if (existingRacks.length > 0) {
                // Position new rack to the right of the rightmost rack
                let maxX = -Infinity;
                existingRacks.forEach(function (rack) {
                    const rackPos = rack.position();
                    if (rackPos.x > maxX) {
                        maxX = rackPos.x;
                        newRackY = rackPos.y; // Use same Y as existing racks
                    }
                });
                newRackX = maxX + 500; // 500px spacing between racks (450 width + 50 spacing)
            } else {
                // First rack - place at a reasonable starting position
                newRackX = LAYOUT_CONSTANTS.NEW_RACK_DEFAULT_X;
                newRackY = LAYOUT_CONSTANTS.NEW_RACK_DEFAULT_Y;
            }

            cy.add({
                group: 'nodes',
                data: {
                    id: newRackId,
                    label: rackLabel,
                    type: 'rack',
                    rack_num: currentRack,
                    hall: currentHall,
                    aisle: currentAisle
                },
                classes: 'rack',
                position: {
                    x: newRackX,
                    y: newRackY
                }
            });
            newRackNode = cy.getElementById(newRackId);
        } else {
            // Update existing rack's label and data
            if (currentHall && currentAisle) {
                newRackNode.data('label', `Rack ${currentRack}`);
                newRackNode.data('hall', currentHall);
                newRackNode.data('aisle', currentAisle);
            }
        }

        // Move the shelf node to the new rack
        node.move({ parent: newRackId });

        // Update all child nodes (trays and ports) with the new rack_num
        node.descendants().forEach(function (child) {
            if (rackChanged) child.data('rack_num', currentRack);
            if (hallChanged && currentHall) child.data('hall', currentHall);
            if (aisleChanged && currentAisle) child.data('aisle', currentAisle);
        });
    } else if (parent && parent.length > 0 && parent.data('type') === 'rack') {
        // Just update parent rack label if hall/aisle changed (but not rack number)
        if (hallChanged || aisleChanged) {
            const rackNum = node.data('rack_num');
            const hall = node.data('hall');
            const aisle = node.data('aisle');

            if (hall && aisle && rackNum !== undefined) {
                parent.data('label', `Rack ${rackNum}`);
                parent.data('hall', hall);
                parent.data('aisle', aisle);
            }

            // Update all child nodes (trays and ports) with the new hall/aisle
            node.descendants().forEach(function (child) {
                if (hallChanged && currentHall) child.data('hall', currentHall);
                if (aisleChanged && currentAisle) child.data('aisle', currentAisle);
            });
        }
    }

    // If rack or shelf_u changed, automatically reset layout to properly reposition nodes
    if (rackChanged || shelfUChanged) {
        // Hide the editing interface first
        hideNodeInfo();

        // Call resetLayout to recalculate positions
        // Use setTimeout to ensure the DOM updates are complete
        setTimeout(function () {
            resetLayout();
        }, 100);
    } else {
        // Hide the editing interface
        hideNodeInfo();

        // Show success message
        showExportStatus('Shelf node updated successfully', 'success');
    }
};

window.cancelShelfEdit = function () {
    hideNodeInfo();
};

function parseConnectionRangeString(rangeString) {
    // Parse a range string like "1-3,5,7-10" into an array of connection numbers
    if (!rangeString || rangeString.trim() === '') {
        return [];
    }

    const connectionNumbers = new Set();
    const parts = rangeString.split(',');

    for (let part of parts) {
        part = part.trim();

        if (part === '') continue;

        if (part.includes('-')) {
            // Handle range (e.g., "1-3" or "7-10")
            const rangeParts = part.split('-');
            if (rangeParts.length !== 2) {
                throw new Error(`Invalid range format: "${part}". Use format like "1-3".`);
            }

            const start = parseInt(rangeParts[0].trim());
            const end = parseInt(rangeParts[1].trim());

            if (isNaN(start) || isNaN(end)) {
                throw new Error(`Invalid numbers in range: "${part}". Both start and end must be numbers.`);
            }

            if (start < 1 || end < 1) {
                throw new Error(`Connection numbers must be at least 1: "${part}".`);
            }

            if (start > end) {
                throw new Error(`Invalid range: "${part}". Start must be less than or equal to end.`);
            }

            // Add all numbers in the range
            for (let i = start; i <= end; i++) {
                connectionNumbers.add(i);
            }
        } else {
            // Handle individual number (e.g., "5")
            const num = parseInt(part);
            if (isNaN(num)) {
                throw new Error(`Invalid number: "${part}". Must be a valid integer.`);
            }

            if (num < 1) {
                throw new Error(`Connection number must be at least 1: "${part}".`);
            }

            connectionNumbers.add(num);
        }
    }

    return Array.from(connectionNumbers).sort((a, b) => a - b);
}

function applyConnectionRange() {
    if (typeof cy === 'undefined' || !cy) {
        showError('Please upload and generate a visualization first.');
        return;
    }

    const rangeInput = document.getElementById('connectionRangeString');
    const statusDiv = document.getElementById('rangeStatus');
    const rangeString = rangeInput.value.trim();

    // If empty, show all connections
    if (rangeString === '') {
        clearConnectionRange();
        return;
    }

    let connectionNumbers;
    try {
        connectionNumbers = parseConnectionRangeString(rangeString);
    } catch (error) {
        showError(`Range format error: ${error.message}`);
        return;
    }

    if (connectionNumbers.length === 0) {
        showError('No valid connection numbers found in range string.');
        return;
    }

    // Get all edges
    const allEdges = cy.edges();
    const totalConnections = allEdges.length;

    // Check if any requested connections exceed total
    const maxRequested = Math.max(...connectionNumbers);
    if (maxRequested > totalConnections) {
        showError(`Connection number ${maxRequested} exceeds total connections (${totalConnections}).`);
        return;
    }

    // Convert array to Set for faster lookup
    const connectionSet = new Set(connectionNumbers);
    let visibleCount = 0;
    let hiddenCount = 0;

    // Get connection type filter settings
    const showIntraNode = document.getElementById('showIntraNodeConnections').checked;
    const showIntraRack = document.getElementById('showIntraRackConnections').checked;
    const showInterRack = document.getElementById('showInterRackConnections').checked;

    // Get node filter setting
    const nodeFilterSelect = document.getElementById('nodeFilterSelect');
    const selectedNodeId = nodeFilterSelect ? nodeFilterSelect.value : '';

    // Filter edges based on connection number, connection type, and node filter
    allEdges.forEach(function (edge) {
        const connectionNumber = parseInt(edge.data('connection_number')) || 0;
        const isIntraNode = edge.data('is_intra_host') === true;
        const sourceRack = edge.source().data('rack_num') || 0;
        const targetRack = edge.target().data('rack_num') || 0;
        const isIntraRack = sourceRack === targetRack && sourceRack > 0;
        const isInterRack = sourceRack !== targetRack && sourceRack > 0 && targetRack > 0;

        // Check if connection should be visible based on type
        let shouldShowByType = false;
        if (isIntraNode && showIntraNode) shouldShowByType = true;
        if (isIntraRack && showIntraRack) shouldShowByType = true;
        if (isInterRack && showInterRack) shouldShowByType = true;

        // Check if connection should be visible based on node filter
        let shouldShowByNode = true;
        if (selectedNodeId !== '') {
            const sourceId = edge.source().id();
            const targetId = edge.target().id();
            shouldShowByNode = (sourceId === selectedNodeId || targetId === selectedNodeId);
        }

        // Show if it matches range, type, and node filters
        if (connectionSet.has(connectionNumber) && shouldShowByType && shouldShowByNode) {
            edge.style('display', 'element');
            visibleCount++;
        } else {
            edge.style('display', 'none');
            hiddenCount++;
        }
    });

    // Create a readable summary of the range
    const rangeSummary = createRangeSummary(connectionNumbers);

    statusDiv.textContent = `Showing connections: ${rangeSummary} (${visibleCount} visible, ${hiddenCount} hidden)`;
    statusDiv.style.color = '#28a745';
}

function createRangeSummary(numbers) {
    // Create a human-readable summary of connection numbers
    if (numbers.length === 0) return 'none';
    if (numbers.length === 1) return numbers[0].toString();

    // Group consecutive numbers into ranges
    const ranges = [];
    let start = numbers[0];
    let end = numbers[0];

    for (let i = 1; i < numbers.length; i++) {
        if (numbers[i] === end + 1) {
            // Consecutive number, extend the range
            end = numbers[i];
        } else {
            // Gap found, close current range and start new one
            if (start === end) {
                ranges.push(start.toString());
            } else {
                ranges.push(`${start}-${end}`);
            }
            start = end = numbers[i];
        }
    }

    // Add the final range
    if (start === end) {
        ranges.push(start.toString());
    } else {
        ranges.push(`${start}-${end}`);
    }

    return ranges.join(', ');
}

function clearConnectionRange() {
    if (typeof cy === 'undefined' || !cy) {
        return;
    }

    // Show all edges
    const allEdges = cy.edges();
    allEdges.forEach(function (edge) {
        edge.style('display', 'element');
    });

    // Clear input value
    document.getElementById('connectionRangeString').value = '';

    // Update status
    const statusDiv = document.getElementById('rangeStatus');
    statusDiv.textContent = `Showing all connections (${allEdges.length} total)`;
    statusDiv.style.color = '#666';
}

/**
 * Validate that all shelf nodes have hostnames
 * @returns {Array} Array of node labels that are missing hostnames (empty if all valid)
 */
function validateHostnames() {
    const nodesWithoutHostname = [];
    cy.nodes().forEach(function (node) {
        const data = node.data();
        if (data.type === 'shelf' && (!data.hostname || data.hostname.trim() === '')) {
            nodesWithoutHostname.push(data.label || data.id);
        }
    });
    return nodesWithoutHostname;
}

async function exportCablingDescriptor() {
    if (typeof cy === 'undefined' || !cy) {
        showExportStatus('No visualization data available', 'error');
        return;
    }

    const exportBtn = document.getElementById('exportCablingBtn');
    const originalText = exportBtn.textContent;

    try {
        exportBtn.textContent = '⏳ Exporting...';
        exportBtn.disabled = true;
        showExportStatus('Generating CablingDescriptor...', 'info');

        // Get current cytoscape data
        const cytoscapeData = {
            elements: cy.elements().jsons()
        };

        // Send to server for processing
        const response = await fetch('/export_cabling_descriptor', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(cytoscapeData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Export failed');
        }

        // Get the textproto content
        const textprotoContent = await response.text();

        // Create and download file
        const blob = new Blob([textprotoContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Use custom filename if provided, otherwise use default
        const customFileName = document.getElementById('exportFileNameInput').value.trim();
        if (customFileName) {
            a.download = `${customFileName}_cabling_descriptor.textproto`;
        } else {
            a.download = 'cabling_descriptor.textproto';
        }

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showExportStatus('CablingDescriptor exported successfully!', 'success');

    } catch (error) {
        console.error('Export error:', error);
        showExportStatus(`Export failed: ${error.message}`, 'error');
    } finally {
        exportBtn.textContent = originalText;
        exportBtn.disabled = false;
    }
}

async function exportDeploymentDescriptor() {
    if (typeof cy === 'undefined' || !cy) {
        showExportStatus('No visualization data available', 'error');
        return;
    }

    const exportBtn = document.getElementById('exportDeploymentBtn');
    const originalText = exportBtn.textContent;

    try {
        exportBtn.textContent = '⏳ Exporting...';
        exportBtn.disabled = true;
        showExportStatus('Generating DeploymentDescriptor...', 'info');

        // Get current cytoscape data
        const cytoscapeData = {
            elements: cy.elements().jsons()
        };

        // Send to server for processing
        const response = await fetch('/export_deployment_descriptor', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(cytoscapeData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Export failed');
        }

        // Get the textproto content
        const textprotoContent = await response.text();

        // Create and download file
        const blob = new Blob([textprotoContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Use custom filename if provided, otherwise use default
        const customFileName = document.getElementById('exportFileNameInput').value.trim();
        if (customFileName) {
            a.download = `${customFileName}_deployment_descriptor.textproto`;
        } else {
            a.download = 'deployment_descriptor.textproto';
        }

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showExportStatus('DeploymentDescriptor exported successfully!', 'success');

    } catch (error) {
        console.error('Export error:', error);
        showExportStatus(`Export failed: ${error.message}`, 'error');
    } finally {
        exportBtn.textContent = originalText;
        exportBtn.disabled = false;
    }
}

async function generateCablingGuide() {
    if (typeof cy === 'undefined' || !cy) {
        showExportStatus('No visualization data available', 'error');
        return;
    }

    // Check for nodes without hostnames and show warning
    const nodesWithoutHostname = validateHostnames();
    if (nodesWithoutHostname.length > 0) {
        showExportStatus(`Warning: The following nodes are missing hostnames: ${nodesWithoutHostname.join(', ')}. FSD generation will proceed but may have incomplete data.`, 'warning');
    }

    const generateBtn = document.getElementById('generateCablingGuideBtn');
    const originalText = generateBtn.textContent;

    try {
        generateBtn.textContent = '⏳ Generating...';
        generateBtn.disabled = true;
        showExportStatus('Generating cabling guide and FSD...', 'info');

        // Get current cytoscape data
        const cytoscapeData = {
            elements: cy.elements().jsons()
        };

        // Get input prefix for the generator
        const customFileName = document.getElementById('exportFileNameInput').value.trim();
        const inputPrefix = customFileName || 'network_topology';

        // Send to server for processing
        const response = await fetch('/generate_cabling_guide', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                cytoscape_data: cytoscapeData,
                input_prefix: inputPrefix
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Generation failed');
        }

        const result = await response.json();

        if (result.success) {
            // Download the generated files directly from content
            if (result.cabling_guide_content) {
                const blob1 = new Blob([result.cabling_guide_content], { type: 'text/csv' });
                const url1 = window.URL.createObjectURL(blob1);
                const a1 = document.createElement('a');
                a1.href = url1;
                a1.download = result.cabling_guide_filename;
                a1.style.display = 'none';
                document.body.appendChild(a1);
                a1.click();
                document.body.removeChild(a1);
                window.URL.revokeObjectURL(url1);
            } else {
            }

            if (result.fsd_content) {
                const blob2 = new Blob([result.fsd_content], { type: 'text/plain' });
                const url2 = window.URL.createObjectURL(blob2);
                const a2 = document.createElement('a');
                a2.href = url2;
                a2.download = result.fsd_filename;
                a2.style.display = 'none';
                document.body.appendChild(a2);
                a2.click();
                document.body.removeChild(a2);
                window.URL.revokeObjectURL(url2);
            } else {
            }

            showExportStatus('Cabling guide and FSD generated successfully!', 'success');
        } else {
            throw new Error(result.error || 'Unknown error occurred');
        }

    } catch (error) {
        console.error('Generation error:', error);
        showExportStatus(`Generation failed: ${error.message}`, 'error');
    } finally {
        generateBtn.textContent = originalText;
        generateBtn.disabled = false;
    }
}

function showExportStatus(message, type) {
    const statusDiv = document.getElementById('exportStatus');
    statusDiv.style.display = 'block';
    statusDiv.textContent = message;

    // Set colors based on type
    if (type === 'success') {
        statusDiv.style.backgroundColor = '#d4edda';
        statusDiv.style.color = '#155724';
        statusDiv.style.border = '1px solid #c3e6cb';
    } else if (type === 'error') {
        statusDiv.style.backgroundColor = '#f8d7da';
        statusDiv.style.color = '#721c24';
        statusDiv.style.border = '1px solid #f5c6cb';
    } else {
        statusDiv.style.backgroundColor = '#d1ecf1';
        statusDiv.style.color = '#0c5460';
        statusDiv.style.border = '1px solid #bee5eb';
    }

    // Auto-hide after 5 seconds for success/info messages
    if (type !== 'error') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
}

// Add keyboard shortcuts for range filtering
document.addEventListener('keydown', function (event) {
    // Ctrl+Enter or Cmd+Enter to apply range
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const rangeInput = document.getElementById('connectionRangeString');

        // Check if range input is focused
        if (document.activeElement === rangeInput) {
            event.preventDefault();
            applyConnectionRange();
        }
    }

    // Escape to clear range
    if (event.key === 'Escape') {
        const rangeInput = document.getElementById('connectionRangeString');

        if (document.activeElement === rangeInput) {
            event.preventDefault();
            clearConnectionRange();
        }
    }

    // Backspace or Delete to delete selected connection (only in editing mode)
    if (event.key === 'Backspace' || event.key === 'Delete') {
        // Only delete if we're in editing mode, have a selected connection, and not typing in an input
        if (isEdgeCreationMode && selectedConnection && selectedConnection.length > 0 &&
            !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
            event.preventDefault();
            deleteSelectedConnection();
        }
    }

    // Ctrl+N to focus on new node hostname input
    if ((event.ctrlKey || event.metaKey) && event.key === 'n') {
        event.preventDefault();
        const hostnameInput = document.getElementById('nodeHostnameInput');
        if (hostnameInput) {
            hostnameInput.focus();
            hostnameInput.select();
        }
    }

    // Ctrl+E to create empty visualization
    if ((event.ctrlKey || event.metaKey) && event.key === 'e') {
        event.preventDefault();
        createEmptyVisualization();
    }
});

// Initialize node configurations when the page loads
initializeNodeConfigs();
