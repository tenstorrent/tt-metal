



Search: syed







Home
1

DMs
2

Activity
3

Files
4

Later
5

More
0

Tenstorrent


More unreads







Messages

CanvasListFolder
10:44
this is the prompt. You'll need to go through it and update it accordingly.
You need to edit ## Prompt to send to the agent and send the part from there and below
SAFETY_COMPONENT_PROMPT_TEMPLATE.md
 

SAFETY_COMPONENT_PROMPT_TEMPLATE.md
Markdown
Safety Component Definition — Agent Prompt Template
Copy this prompt and fill in the <USER_INPUT> placeholders before sending it to an agent.

User inputs (fill these in)
Component name: <COMPONENT_NAME> — full name of the software item (e.g., System Health Monitor).
Component abbreviation: <COMP> — short prefix for requirement IDs (e.g., SHM).
Repository root: <REPO_ROOT> — absolute or repo-relative path to the repository (e.g., /Users/snijjar/work/tt-metal).
Component root folder: <COMPONENT_ROOT> — path to the top-level source folder for this component (e.g., tt-metal/tt_metal/impl/health_monitor).
Template file path: <TEMPLATE_PATH> — path to the generic safety component definition template. Default: <REPO_ROOT>/tests/tt_metal/tt_fabric/fabric_data_movement/Template.
Output directory: <OUTPUT_DIR> — where the agent should write the populated files (e.g., <REPO_ROOT>/safety/<comp>/).
Reference safety README URL: <SAFETY_README_URL> — program-level reference README (optional).
ASIL target: <ASIL_TARGET> — default ASIL-D.
Known product requirements / specs: <PRODUCT_REQS> — optional links or text to product requirements, architecture docs, or formal specs.
Known neighbor domains / ownership: <NEIGHBORS> — optional list of neighboring domains (e.g., Runtime, Safety Manager, UMD, HAL) and any known ownership boundaries.
Known open questions / decisions: <OPEN_DECISIONS> — optional list of open items the agent should resolve or track.
Scale / topology / envelope limits: <ENVELOPE> — optional known supported configuration limits.
Prompt to send to the agent
You are a functional-safety agent working on an AI-IP SW SEooC baseline. Your job is to
populate a component safety item definition and derive first-cut safety requirements for a
software item targeted at ASIL-D.

## Inputs

- Component name: <COMPONENT_NAME>
- Component abbreviation: <COMP>
- Repository root: <REPO_ROOT>
- Component root folder: <COMPONENT_ROOT>
- Template file: <TEMPLATE_PATH>
- Output directory: <OUTPUT_DIR>
- Reference safety README: <SAFETY_README_URL>
- ASIL target: <ASIL_TARGET>
- Product requirements / specs (if any): <PRODUCT_REQS>
- Neighbor domains / ownership (if any): <NEIGHBORS>
- Known open questions / decisions (if any): <OPEN_DECISIONS>
- Supported envelope limits (if any): <ENVELOPE>

## Instructions

1. Read the template file at `<TEMPLATE_PATH>`. Understand every section, the
   `<ANGLE_BRACKETED>` placeholders, the agent instructions, and the worked example.

2. If `<SAFETY_README_URL>` is provided, use the `gh` CLI (already installed and
   authenticated) to fetch the latest reference safety README and any linked files you need
   for style. Use commands like:

   gh api repos/<org>/<repo>/contents/<path> --jq '.content' | base64 -d

   If no reference URL is provided, proceed using the template alone and do not assume a
   program-specific style beyond what is documented in the template.

3. Explore the component codebase under `<COMPONENT_ROOT>` thoroughly. Identify:
   - public APIs, control surfaces, and entry points
   - internal functions and subfunctions
   - configuration / descriptor / metadata handling
   - upward, downward, and environmental interfaces
   - lifecycle behavior (initialization, runtime, teardown)
   - error / fault / status / completion reporting paths
   - safety-relevant code paths and existing invariants
   - performance limits or supported envelope where documented

4. Populate every section of the template, replacing all placeholders with
   component-specific content. Every claim must be traceable to a code artifact, header,
   API contract, or architecture decision. Keep variant-agnostic descriptions in the main
   sections; move variant-specific assumptions into the "Assumptions of use" section.

5. Include UML diagrams in Section 16 of the item definition where they clarify the
   boundary, interfaces, or behavior. At minimum provide a boundary / context diagram.
   Internal function, layered component, and functional-flow diagrams are optional but
   strongly encouraged where they make sense.

6. Produce the following output files in `<OUTPUT_DIR>`:
   - `<COMP>-item-definition.md` — the fully populated template (Sections 1–17)
   - `<COMP>-safety-properties.md` — negative / invariant requirements (what must never go wrong)
   - `<COMP>-functional-capabilities.md` — positive / measurable requirements + performance + envelope
   - `<COMP>-functions.md` — function / subfunction decomposition with ownership
   - `<COMP>-failure-considerations.md` — preliminary malfunction list mapped to functions / requirements
   - `<COMP>-assumptions-pre-post-deps.md` — assumptions, preconditions, postconditions, dependencies
   - `<COMP>-boundary-and-interfaces.md` — diagrams + interface specification (upward / downward / environmental)
   - `<COMP>-baseline-decisions.md` — open decisions, owner assignments, and resolution status

7. Follow the requirement conventions from the template:
   - Provenance tags: `SPEC:<id>`, `CODE:<path>`, `HAZARD:§13`, `PRODUCT`
   - Status tags: `FIRM`, `CANDIDATE`, `PROPOSED`
   - Safety-property IDs: `<COMP>-HLR-<nn>` grouped under Safety Goals `SG-<X>`
   - Functional-capability IDs: `<COMP>-FR-<nn>`, performance `<COMP>-PERF-<nn>`, envelope `<COMP>-ENV-<nn>`

8. For every safety requirement, provide:
   - a unique, prefixed ID
   - one atomic shall-statement (avoid ambiguous "and" / "or")
   - the ASIL target
   - a rationale linking the requirement to a failure theme, hazard, or safety goal
   - allocation to the item or a neighboring item / external safety mechanism
   - verification method(s): Test, Review, Analysis, Walk-through, Fault injection, Simulation, Static analysis

9. Do not invent product requirements or capabilities that are not present in the code or
   supplied product requirements. If something is required by a §13 failure theme but not
   implemented or specified, mark it `PROPOSED` and record it as a gap in
   `<COMP>-baseline-decisions.md`.

10. If you encounter ownership questions that cannot be resolved from the code, record them
    in `<COMP>-baseline-decisions.md` with a proposed owner and supporting evidence. Do not
    guess ownership to close the gap.

11. Update the baselining status checklist (Section 17.5 of the item definition) to reflect
    which steps are complete. Mark completed items with `[x]` and remaining items with `[ ]`.

12. Do not run tests, build the project, or commit/push changes unless explicitly asked.

## Deliverables

- All files written to `<OUTPUT_DIR>`
- A brief summary of what was populated, any gaps found, any `PROPOSED` requirements, and any
  questions that need human resolution before baselining
Notes for the human operator
The default template path points to the generic ASIL-D template created in the Fabric directory. If you move the template, update <TEMPLATE_PATH> accordingly.
The reference README is private; the agent should already be authenticated to gh in the environment. If not, the agent will need to run gh auth login or use a token.
If you have formal specs, product requirements, or an existing hazard list, include them in <PRODUCT_REQS> so the agent can tag requirements with SPEC: or HAZARD: provenance.
The more specific the <NEIGHBORS> and <OPEN_DECISIONS> inputs are, the fewer ownership questions the agent will need to escalate.



10:47
This is the document structure guide for the agent. I derived it from @Ashish Jogeshwar’s original docs and added additional information about how the content should be structured and written. Consider updating it for additional guidance you feel is necessary for your use case.
I haven't tried getting the agent to take this and the confluence API token and generate directly into the doc (I suspect it would work quite well) - it might be worth trying that :slightly_smiling_face: When I did it originally I needed a translation layer to copy into the confluence page -- you can probably skip a step.
document_template.md
 

document_template.md
Markdown
AI-IP SW Safety Component Definition Template ASIL-D / SEooC — Item Definition and Preliminary Requirements

How to use this template
Replace every <ANGLE_BRACKETED> placeholder with component-specific text.
Base all content on the actual source code, header files, API contracts, and design documents of the component under analysis.
Keep descriptions variant-agnostic at this stage; move variant-specific assumptions into the “Assumptions of use” section.
Every claim about what the component does or does not do must be traceable to a code artifact, interface, or architecture decision.
Use ASIL-D language throughout: safety requirements, safety mechanisms, fault avoidance, fault detection, fault containment, safe state, FTTI, independence, etc.
Follow the requirement split in Section 17: safety properties (negative / invariant) and functional capabilities (positive / measurable). Use the provenance tags SPEC:, CODE:, HAZARD:, and PRODUCT and status tags FIRM, CANDIDATE, PROPOSED.
Sections marked with a non-Fabric example are illustrative only. Remove the example text when populating the template for a real component, or keep it as a reference in a draft until the page is baselined.
Include UML diagrams in Section 16 where they clarify the boundary, interfaces, or behavior. Diagrams are appreciated where they make sense; a boundary / context diagram is expected, others are optional but encouraged.
Primary reference notes
Program-level safety reference: <PROGRAM_LEVEL_SAFETY_README_URL>

The safety work is structured as:

Two requirement files: safety-properties.md (what must never go wrong) and functional-capabilities.md (what the item must do, plus performance/envelope).
Provenance tags: SPEC:<id> (spec-backed), CODE:<path> (impl-observed), HAZARD:§13 (top-down failure-theme), PRODUCT (product requirement).
Status tags: FIRM (spec-backed), CANDIDATE (code-derived, confirm), PROPOSED (hazard-derived, decide ownership).
Baselining steps A–F (item definition → boundary → interfaces → functions → failure considerations → assumptions/pre/post/deps).
Replace <PROGRAM_LEVEL_SAFETY_README_URL> with the component- or program-level safety README applicable to this item.

Item name ============
AGENT INSTRUCTIONS

State the concise, unambiguous name of the software item.
Use the same name as in the architecture, module boundary, or source tree.
PURPOSE

Provides a unique identifier for the safety case, traceability, and work-product cross-references.
ASIL-D GUIDANCE

The item name is the root of the safety requirement IDs and of the DFA / FMEDA traceability.
TEMPLATE <COMPONENT_NAME>

EXAMPLE (non-Fabric) System Health Monitor (SHM)

Item purpose ===============
AGENT INSTRUCTIONS

Describe what the software item does from a functional, safety-relevant perspective.
Avoid implementation details; focus on the intent the item fulfills for the rest of the AI-IP SW stack.
Identify the safety-relevant services the item provides (e.g., data integrity, error detection, timing protection, access control, health monitoring).
PURPOSE

Establishes the “why” of the item and is the basis for deriving functional safety requirements.
ASIL-D GUIDANCE

At ASIL-D, the purpose must be precise enough to support hazard analysis and malfunctional analysis; every safety requirement must be derivable from this purpose.
State which safety goals the item contributes to, if known.
TEMPLATE <COMPONENT_NAME> provides the <PRIMARY_CAPABILITY> used to support <SYSTEM_FUNCTION> across the AI-IP system. In the current safety-baseline framing, it should be treated as a software item whose purpose is to enable <BEHAVIOR> through defined software-controlled interfaces and configuration paths.

Safety-relevant aspects currently allocated to <COMPONENT_NAME> include:

<SAFETY_RELEVANT_SERVICE_1>
<SAFETY_RELEVANT_SERVICE_2>
<SAFETY_RELEVANT_SERVICE_3>
At this stage, the purpose statement should remain generic across variants and customers, then be refined later for specific <VARIANTS / SCALE / CONFIGURATIONS> as those become controlled assumptions or requirements.

EXAMPLE (non-Fabric) System Health Monitor (SHM) provides the software-visible health supervision, fault aggregation, and safe-state notification services used to support system-wide fault detection and containment across the AI-IP system. In the current safety-baseline framing, it should be treated as a software item whose purpose is to enable timely detection of device-level anomalies and propagation of fault status to higher-level safety monitors through defined software-controlled interfaces and configuration paths.

Safety-relevant aspects currently allocated to SHM include:

periodic collection of health indicators from hardware monitors
detection of out-of-range or missing telemetry
notification of fault conditions to the safety manager
Item type ============
AGENT INSTRUCTIONS

Classify the item (e.g., software item, hardware-software interface, tool, library, service).
If it is a tool, state the tool category per ISO 26262-8 (TCL1–TCL3).
PURPOSE

Drives which ISO 26262 / ASPICE work products are required.
ASIL-D GUIDANCE

ASIL-D demands the highest rigor for software items; tool qualification may be required if the item generates code or safety-relevant data.
TEMPLATE <SOFTWARE_ITEM | HARDWARE_SOFTWARE_INTERFACE | TOOL (TCLx) | LIBRARY | SERVICE> within the AI-IP SW stack.

EXAMPLE (non-Fabric) Software item within the AI-IP SW stack.

Item context ===============
AGENT INSTRUCTIONS

Explain where the item sits in the program-level safety architecture.
Name neighboring domains and the program-level decomposition approach.
State the current safety program phase (scope / boundary / interface work before requirements).
PURPOSE

Shows how this item contributes to the overall SEooC boundary.
ASIL-D GUIDANCE

Context must be consistent with the SEooC safety plan and the assumed product-level safety architecture.
TEMPLATE The AI-IP SW safety program baseline explicitly includes <COMPONENT_NAME> as one of the software functional areas to be defined within the cross-program SEooC boundary, alongside domains such as <DOMAIN_1>, <DOMAIN_2>, and <DOMAIN_3>.

The current working process for the safety baseline is to start each domain with scope, boundary, interface, and assumptions work before deriving requirements, failure analysis, and verification planning. <COMPONENT_NAME> is being handled through that domain-by-domain item-definition approach.

Within that context, <COMPONENT_NAME> should be positioned as a <DOMAIN_ORIENTATION> software domain whose exact decomposition and supported behaviors will later be specialized by variant, customer scope, and architecture baseline.

EXAMPLE (non-Fabric) The AI-IP SW safety program baseline explicitly includes System Health Monitor as one of the software functional areas to be defined within the cross-program SEooC boundary, alongside domains such as UMD, Runtime, Fabric, and TTNN / Kernel Ops.

The current working process for the safety baseline is to start each domain with scope, boundary, interface, and assumptions work before deriving requirements, failure analysis, and verification planning. System Health Monitor is being handled through that domain-by-domain item-definition approach.

Within that context, System Health Monitor should be positioned as a system-level supervision-oriented software domain whose exact decomposition and supported behaviors will later be specialized by variant, customer scope, and architecture baseline.

Item boundary ================
AGENT INSTRUCTIONS

Define the coarse and detailed boundaries of the item.
List what is inside the item (allocated to this software item) and what is outside (allocated to other items or assumed).
Be precise enough that another engineer could draw a boundary diagram.
Include UML diagrams where they help clarify the boundary (context, internal function, layered component, functional flow). Diagrams are appreciated where they make sense.
Place diagrams in Section 16 and reference them from the boundary sections.
PURPOSE

Establishes the SEooC interface and scope for safety analysis.
ASIL-D GUIDANCE

Boundaries must be unambiguous; unclear boundaries are a major source of ASIL-D audit findings.
Everything outside the boundary must be covered by an assumption of use or a dependency.
5.1 Coarse boundary
AGENT INSTRUCTIONS

Provide a one-paragraph summary of the item boundary at a high level.
This is the boundary you would show on a context diagram.
Provide a PlantUML context diagram in Section 16 if it clarifies the scope.
TEMPLATE <ONE_PARAGRAPH_SUMMARY>

EXAMPLE (non-Fabric) The SHM boundary covers the software that collects, validates, and reports device health indicators to higher-level safety consumers. It does not include the physical sensors, the safety manager’s fault reaction decisions, or the hardware reset mechanisms.

5.2 Detailed boundary
AGENT INSTRUCTIONS

Provide a more granular boundary statement, e.g., file paths, API groups, classes, or kernel modules.
This can be a list or a reference to the architecture document.
Include a more detailed boundary or package diagram if it helps distinguish this item from neighboring domains.
TEMPLATE <DETAILED_BOUNDARY_LIST_OR_REFERENCE>

EXAMPLE (non-Fabric) Inside the detailed boundary: SHM host service, SHM device kernel, health-indicator abstraction layer, threshold manager, alarm dispatcher, and diagnostic telemetry hooks. Outside: hardware sensor drivers in the HAL, safety manager arbitration logic, and reset/safe-state actuation hardware.

5.3 Inside the item
AGENT INSTRUCTIONS

List all software behavior, data structures, configuration, and lifecycle behavior owned by this item.
TEMPLATE The following should be considered inside the current high-level <COMPONENT_NAME> item boundary:

<COMPONENT_NAME> software behavior that defines <PRIMARY_BEHAVIOR>
<AWARENESS_TYPE> software behavior allocated to <COMPONENT_NAME>
software-visible control, coordination, or <REQUEST_TYPE> handling allocated to <COMPONENT_NAME>; includes <SAFETY_BEHAVIOR_1>
<COMPONENT_NAME>-specific configuration handling, descriptors, or metadata handling where allocated to this software item, including <REQUEST_CONSTRUCTION>
status, error, or completion reporting behavior that is owned by <COMPONENT_NAME>
software abstractions used to request or coordinate <COORDINATION_DOMAIN> across connected compute or device elements
lifecycle management: <LIFECYCLE_PHASES> (potential to move to tool if applicable)
EXAMPLE (non-Fabric) The following should be considered inside the current high-level System Health Monitor item boundary:

SHM software behavior that defines supported health monitoring and fault notification intent
anomaly-detection or threshold-aware software behavior allocated to SHM
software-visible control, polling, or interrupt-driven handling allocated to SHM; includes fault-firing behavior
SHM-specific configuration handling, thresholds, or metadata handling where allocated to this software item, including alarm construction and dispatch
status, error, or completion reporting behavior that is owned by SHM
software abstractions used to request or coordinate health checks across connected compute or device elements
lifecycle management: initialization / runtime monitoring / teardown behaviour (potential to move to tool)
5.4 Outside the item
AGENT INSTRUCTIONS

List everything outside the boundary unless explicitly allocated later.
TEMPLATE The following should be considered outside the current high-level <COMPONENT_NAME> item boundary unless explicitly allocated later:

application logic and <POLICY_TYPE> above the <COMPONENT_NAME> software boundary
generic <RUNTIME_TYPE> policy not owned by <COMPONENT_NAME>
lower-level <PLATFORM_TYPE> implementation not owned by <COMPONENT_NAME>
<HARDWARE_TYPE> and physical <PHYSICAL_BEHAVIOR> themselves
system-level safety mechanisms, supervision, or fault-management logic external to the AI-IP SW <COMPONENT_NAME> scope
customer-specific platform integration details unless they are explicitly brought into the supported baseline
<EXCLUDED_DOMAIN_1>
<EXCLUDED_DOMAIN_2>
<EXCLUDED_DOMAIN_3>
This style of defining what is inside versus outside the item is consistent with the structure already used in the item-definition template and with the safety-plan requirement to make the SEooC boundary and assumptions of use explicit.

EXAMPLE (non-Fabric) The following should be considered outside the current high-level System Health Monitor item boundary unless explicitly allocated later:

application-level safety policy and fault reaction decisions above the SHM software boundary
generic runtime policy not owned by SHM
lower-level sensor hardware or ADC/PMU implementation not owned by SHM
hardware sensor, thermal monitor, and physical health indicator behavior themselves
system-level safety …</EXCLUDED_DOMAIN_3></EXCLUDED_DOMAIN_2></EXCLUDED_DOMAIN_1></COMPONENT_NAME></PHYSICAL_BEHAVIOR></HARDWARE_TYPE></COMPONENT_NAME></PLATFORM_TYPE></COMPONENT_NAME></RUNTIME_TYPE></COMPONENT_NAME></POLICY_TYPE></COMPONENT_NAME></LIFECYCLE_PHASES></COORDINATION_DOMAIN></COMPONENT_NAME></REQUEST_CONSTRUCTION></COMPONENT_NAME></SAFETY_BEHAVIOR_1></COMPONENT_NAME></REQUEST_TYPE></COMPONENT_NAME></AWARENESS_TYPE></PRIMARY_BEHAVIOR></COMPONENT_NAME></COMPONENT_NAME></DETAILED_BOUNDARY_LIST_OR_REFERENCE></ONE_PARAGRAPH_SUMMARY></DOMAIN_ORIENTATION></COMPONENT_NAME></COMPONENT_NAME></DOMAIN_3></DOMAIN_2></DOMAIN_1></COMPONENT_NAME></SAFETY_RELEVANT_SERVICE_3></SAFETY_RELEVANT_SERVICE_2></SAFETY_RELEVANT_SERVICE_1></COMPONENT_NAME></BEHAVIOR></SYSTEM_FUNCTION></PRIMARY_CAPABILITY></COMPONENT_NAME></COMPONENT_NAME></PROGRAM_LEVEL_SAFETY_README_URL></path></id></PROGRAM_LEVEL_SAFETY_README_URL></ANGLE_BRACKETED>



Arik Yaacob
  10:52 AM
it's good you had requirement and design documents. Runtime has been in flux ever since it's creation
Sean Nijjar
  10:55 AM
tbh, fabric has been as well but my original motivation was to formalize some specs/arch so we could start moving more dev over to AI. They're pretty bad so they at the very least need some detailed/concrete description but ideally need hard checks (compiler analyses, formal, etc.). Because the specifics change I started with the general invariants. Things like deadlock safety, topology, boundaries with other components etc (which lines up with what we're doing with FUSA).












Message Sean Nijjar:airplane:Disagg f2f sc









Shift + Return to add a new line




AI-IP SW Safety Component Definition Template ASIL-D / SEooC — Item Definition and Preliminary Requirements

How to use this template
Replace every <ANGLE_BRACKETED> placeholder with component-specific text.
Base all content on the actual source code, header files, API contracts, and design documents of the component under analysis.
Keep descriptions variant-agnostic at this stage; move variant-specific assumptions into the “Assumptions of use” section.
Every claim about what the component does or does not do must be traceable to a code artifact, interface, or architecture decision.
Use ASIL-D language throughout: safety requirements, safety mechanisms, fault avoidance, fault detection, fault containment, safe state, FTTI, independence, etc.
Follow the requirement split in Section 17: safety properties (negative / invariant) and functional capabilities (positive / measurable). Use the provenance tags SPEC:, CODE:, HAZARD:, and PRODUCT and status tags FIRM, CANDIDATE, PROPOSED.
Sections marked with a non-Fabric example are illustrative only. Remove the example text when populating the template for a real component, or keep it as a reference in a draft until the page is baselined.
Include UML diagrams in Section 16 where they clarify the boundary, interfaces, or behavior. Diagrams are appreciated where they make sense; a boundary / context diagram is expected, others are optional but encouraged.
Primary reference notes
Program-level safety reference: <PROGRAM_LEVEL_SAFETY_README_URL>

The safety work is structured as:

Two requirement files: safety-properties.md (what must never go wrong) and functional-capabilities.md (what the item must do, plus performance/envelope).
Provenance tags: SPEC:<id> (spec-backed), CODE:<path> (impl-observed), HAZARD:§13 (top-down failure-theme), PRODUCT (product requirement).
Status tags: FIRM (spec-backed), CANDIDATE (code-derived, confirm), PROPOSED (hazard-derived, decide ownership).
Baselining steps A–F (item definition → boundary → interfaces → functions → failure considerations → assumptions/pre/post/deps).
Replace <PROGRAM_LEVEL_SAFETY_README_URL> with the component- or program-level safety README applicable to this item.

Item name ============
AGENT INSTRUCTIONS

State the concise, unambiguous name of the software item.
Use the same name as in the architecture, module boundary, or source tree.
PURPOSE

Provides a unique identifier for the safety case, traceability, and work-product cross-references.
ASIL-D GUIDANCE

The item name is the root of the safety requirement IDs and of the DFA / FMEDA traceability.
TEMPLATE <COMPONENT_NAME>

EXAMPLE (non-Fabric) System Health Monitor (SHM)

Item purpose ===============
AGENT INSTRUCTIONS

Describe what the software item does from a functional, safety-relevant perspective.
Avoid implementation details; focus on the intent the item fulfills for the rest of the AI-IP SW stack.
Identify the safety-relevant services the item provides (e.g., data integrity, error detection, timing protection, access control, health monitoring).
PURPOSE

Establishes the “why” of the item and is the basis for deriving functional safety requirements.
ASIL-D GUIDANCE

At ASIL-D, the purpose must be precise enough to support hazard analysis and malfunctional analysis; every safety requirement must be derivable from this purpose.
State which safety goals the item contributes to, if known.
TEMPLATE <COMPONENT_NAME> provides the <PRIMARY_CAPABILITY> used to support <SYSTEM_FUNCTION> across the AI-IP system. In the current safety-baseline framing, it should be treated as a software item whose purpose is to enable <BEHAVIOR> through defined software-controlled interfaces and configuration paths.

Safety-relevant aspects currently allocated to <COMPONENT_NAME> include:

<SAFETY_RELEVANT_SERVICE_1>
<SAFETY_RELEVANT_SERVICE_2>
<SAFETY_RELEVANT_SERVICE_3>
At this stage, the purpose statement should remain generic across variants and customers, then be refined later for specific <VARIANTS / SCALE / CONFIGURATIONS> as those become controlled assumptions or requirements.

EXAMPLE (non-Fabric) System Health Monitor (SHM) provides the software-visible health supervision, fault aggregation, and safe-state notification services used to support system-wide fault detection and containment across the AI-IP system. In the current safety-baseline framing, it should be treated as a software item whose purpose is to enable timely detection of device-level anomalies and propagation of fault status to higher-level safety monitors through defined software-controlled interfaces and configuration paths.

Safety-relevant aspects currently allocated to SHM include:

periodic collection of health indicators from hardware monitors
detection of out-of-range or missing telemetry
notification of fault conditions to the safety manager
Item type ============
AGENT INSTRUCTIONS

Classify the item (e.g., software item, hardware-software interface, tool, library, service).
If it is a tool, state the tool category per ISO 26262-8 (TCL1–TCL3).
PURPOSE

Drives which ISO 26262 / ASPICE work products are required.
ASIL-D GUIDANCE

ASIL-D demands the highest rigor for software items; tool qualification may be required if the item generates code or safety-relevant data.
TEMPLATE <SOFTWARE_ITEM | HARDWARE_SOFTWARE_INTERFACE | TOOL (TCLx) | LIBRARY | SERVICE> within the AI-IP SW stack.

EXAMPLE (non-Fabric) Software item within the AI-IP SW stack.

Item context ===============
AGENT INSTRUCTIONS

Explain where the item sits in the program-level safety architecture.
Name neighboring domains and the program-level decomposition approach.
State the current safety program phase (scope / boundary / interface work before requirements).
PURPOSE

Shows how this item contributes to the overall SEooC boundary.
ASIL-D GUIDANCE

Context must be consistent with the SEooC safety plan and the assumed product-level safety architecture.
TEMPLATE The AI-IP SW safety program baseline explicitly includes <COMPONENT_NAME> as one of the software functional areas to be defined within the cross-program SEooC boundary, alongside domains such as <DOMAIN_1>, <DOMAIN_2>, and <DOMAIN_3>.

The current working process for the safety baseline is to start each domain with scope, boundary, interface, and assumptions work before deriving requirements, failure analysis, and verification planning. <COMPONENT_NAME> is being handled through that domain-by-domain item-definition approach.

Within that context, <COMPONENT_NAME> should be positioned as a <DOMAIN_ORIENTATION> software domain whose exact decomposition and supported behaviors will later be specialized by variant, customer scope, and architecture baseline.

EXAMPLE (non-Fabric) The AI-IP SW safety program baseline explicitly includes System Health Monitor as one of the software functional areas to be defined within the cross-program SEooC boundary, alongside domains such as UMD, Runtime, Fabric, and TTNN / Kernel Ops.

The current working process for the safety baseline is to start each domain with scope, boundary, interface, and assumptions work before deriving requirements, failure analysis, and verification planning. System Health Monitor is being handled through that domain-by-domain item-definition approach.

Within that context, System Health Monitor should be positioned as a system-level supervision-oriented software domain whose exact decomposition and supported behaviors will later be specialized by variant, customer scope, and architecture baseline.

Item boundary ================
AGENT INSTRUCTIONS

Define the coarse and detailed boundaries of the item.
List what is inside the item (allocated to this software item) and what is outside (allocated to other items or assumed).
Be precise enough that another engineer could draw a boundary diagram.
Include UML diagrams where they help clarify the boundary (context, internal function, layered component, functional flow). Diagrams are appreciated where they make sense.
Place diagrams in Section 16 and reference them from the boundary sections.
PURPOSE

Establishes the SEooC interface and scope for safety analysis.
ASIL-D GUIDANCE

Boundaries must be unambiguous; unclear boundaries are a major source of ASIL-D audit findings.
Everything outside the boundary must be covered by an assumption of use or a dependency.
5.1 Coarse boundary
AGENT INSTRUCTIONS

Provide a one-paragraph summary of the item boundary at a high level.
This is the boundary you would show on a context diagram.
Provide a PlantUML context diagram in Section 16 if it clarifies the scope.
TEMPLATE <ONE_PARAGRAPH_SUMMARY>

EXAMPLE (non-Fabric) The SHM boundary covers the software that collects, validates, and reports device health indicators to higher-level safety consumers. It does not include the physical sensors, the safety manager’s fault reaction decisions, or the hardware reset mechanisms.

5.2 Detailed boundary
AGENT INSTRUCTIONS

Provide a more granular boundary statement, e.g., file paths, API groups, classes, or kernel modules.
This can be a list or a reference to the architecture document.
Include a more detailed boundary or package diagram if it helps distinguish this item from neighboring domains.
TEMPLATE <DETAILED_BOUNDARY_LIST_OR_REFERENCE>

EXAMPLE (non-Fabric) Inside the detailed boundary: SHM host service, SHM device kernel, health-indicator abstraction layer, threshold manager, alarm dispatcher, and diagnostic telemetry hooks. Outside: hardware sensor drivers in the HAL, safety manager arbitration logic, and reset/safe-state actuation hardware.

5.3 Inside the item
AGENT INSTRUCTIONS

List all software behavior, data structures, configuration, and lifecycle behavior owned by this item.
TEMPLATE The following should be considered inside the current high-level <COMPONENT_NAME> item boundary:

<COMPONENT_NAME> software behavior that defines <PRIMARY_BEHAVIOR>
<AWARENESS_TYPE> software behavior allocated to <COMPONENT_NAME>
software-visible control, coordination, or <REQUEST_TYPE> handling allocated to <COMPONENT_NAME>; includes <SAFETY_BEHAVIOR_1>
<COMPONENT_NAME>-specific configuration handling, descriptors, or metadata handling where allocated to this software item, including <REQUEST_CONSTRUCTION>
status, error, or completion reporting behavior that is owned by <COMPONENT_NAME>
software abstractions used to request or coordinate <COORDINATION_DOMAIN> across connected compute or device elements
lifecycle management: <LIFECYCLE_PHASES> (potential to move to tool if applicable)
EXAMPLE (non-Fabric) The following should be considered inside the current high-level System Health Monitor item boundary:

SHM software behavior that defines supported health monitoring and fault notification intent
anomaly-detection or threshold-aware software behavior allocated to SHM
software-visible control, polling, or interrupt-driven handling allocated to SHM; includes fault-firing behavior
SHM-specific configuration handling, thresholds, or metadata handling where allocated to this software item, including alarm construction and dispatch
status, error, or completion reporting behavior that is owned by SHM
software abstractions used to request or coordinate health checks across connected compute or device elements
lifecycle management: initialization / runtime monitoring / teardown behaviour (potential to move to tool)
5.4 Outside the item
AGENT INSTRUCTIONS

List everything outside the boundary unless explicitly allocated later.
TEMPLATE The following should be considered outside the current high-level <COMPONENT_NAME> item boundary unless explicitly allocated later:

application logic and <POLICY_TYPE> above the <COMPONENT_NAME> software boundary
generic <RUNTIME_TYPE> policy not owned by <COMPONENT_NAME>
lower-level <PLATFORM_TYPE> implementation not owned by <COMPONENT_NAME>
<HARDWARE_TYPE> and physical <PHYSICAL_BEHAVIOR> themselves
system-level safety mechanisms, supervision, or fault-management logic external to the AI-IP SW <COMPONENT_NAME> scope
customer-specific platform integration details unless they are explicitly brought into the supported baseline
<EXCLUDED_DOMAIN_1>
<EXCLUDED_DOMAIN_2>
<EXCLUDED_DOMAIN_3>
This style of defining what is inside versus outside the item is consistent with the structure already used in the item-definition template and with the safety-plan requirement to make the SEooC boundary and assumptions of use explicit.

EXAMPLE (non-Fabric) The following should be considered outside the current high-level System Health Monitor item boundary unless explicitly allocated later:

application-level safety policy and fault reaction decisions above the SHM software boundary
generic runtime policy not owned by SHM
lower-level sensor hardware or ADC/PMU implementation not owned by SHM
hardware sensor, thermal monitor, and physical health indicator behavior themselves
system-level safety mechanisms, supervision, or fault-management logic external to the AI-IP SW SHM scope
customer-specific platform integration details unless they are explicitly brought into the supported baseline
safety manager / FTTI arbitration
reset or safe-state actuation controlled by hardware or another domain
watchdog servicing owned by a separate watchdog item
External interfaces ======================
AGENT INSTRUCTIONS

Enumerate upward (consumer), downward (provider), and environmental (context) interfaces.
For each interface, state the interface owner, the data/control exchanged, and the safety-relevance.
PURPOSE

Interfaces are the primary source of safety requirements and are also the focus of dependent failure analysis (DFA).
ASIL-D GUIDANCE

At ASIL-D, every interface must be analyzed for common-cause failures; data and control flow must be independently verifiable where possible.
6.1 Upward interfaces
AGENT INSTRUCTIONS

List higher-level consumers that call this item.
TEMPLATE Potential upward interfaces include:

higher-level host or runtime software that requests supported <BEHAVIOR>
<CONSUMER_1>, <CONSUMER_2>, or similar software consumers if <COMPONENT_NAME> exposes supported APIs or control surfaces to them
higher-level orchestration or <ORCHESTRATION_TYPE> software where such interactions are part of delivered scope
<COMPONENT_NAME> APIs exposed to workloads for usability including <API_CAPABILITY_1>, <API_CAPABILITY_2>, <API_CAPABILITY_3>
EXAMPLE (non-Fabric) Potential upward interfaces include:

higher-level host or runtime software that requests supported health status queries
Runtime, distributed runtime, or safety manager if SHM exposes supported APIs or control surfaces to them
higher-level orchestration or fault-management software where such interactions are part of delivered scope
SHM APIs exposed to workloads for usability including health query, threshold configuration, and alarm subscription
6.2 Downward interfaces
AGENT INSTRUCTIONS

List lower-level providers this item uses.
TEMPLATE Potential downward interfaces include:

Host side (tool candidate if applicable):

<HOST_SERVICE_1>: <PURPOSE>
<HOST_SERVICE_2>: <PURPOSE>
Device side:

device-level or <LOWER_BOUNDARY_TYPE> software/hardware boundaries used to carry out requested <ACTION>
<LOWER_API_1>
physical <TRANSPORT_API> (e.g., <EXAMPLE_TRANSPORT>)
kernel or driver boundary services where <COMPONENT_NAME> depends on lower-layer <LOWER_CAPABILITY> or control mechanisms
lower software layers that implement the actual <EXECUTION_PATH> execution path
EXAMPLE (non-Fabric) Potential downward interfaces include:

Host side (tool candidate):

HAL: device register addresses, sensor locations, etc.
UMD: register read/write, device status
Device side:

device-level or hardware-monitor-facing software/hardware boundaries used to carry out requested health sampling
sensor read APIs
interrupt aggregation APIs
kernel or driver boundary services where SHM depends on lower-layer register access or control mechanisms
lower software layers that implement the actual sampling execution path
6.3 Environmental interfaces
AGENT INSTRUCTIONS

List environmental/context dependencies (topology, configuration, hardware, power, etc.).
TEMPLATE Potential environmental interfaces include:

<CONFIG_TYPE> description and coordinate-system information
<DESCRIPTOR_TYPE> descriptor
device discovery / enumeration information
<HARDWARE_TYPE> and inter-device connectivity assumptions exposed to software
monitoring, diagnostics, telemetry, timeout, or fault-reporting channels relevant to <COMPONENT_NAME> behavior
reset, power, and initialization state dependencies where these constrain supported <COMPONENT_NAME> operation
The safety plan expects interface requirements to be defined across domains including <COMPONENT_NAME>, and the current template structure also separates upward, downward, and environmental interfaces, so this interface split is a good fit for the item-definition workflow.

EXAMPLE (non-Fabric) Potential environmental interfaces include:

device topology description and coordinate-system information
sensor map descriptor
device discovery / enumeration information
thermal sensor and health-monitor connectivity assumptions exposed to software
monitoring, diagnostics, telemetry, timeout, or fault-reporting channels relevant to SHM behavior
reset, power, and initialization state dependencies where these constrain supported SHM operation
The safety plan expects interface requirements to be defined across domains including System Health Monitor, and the current template structure also separates upward, downward, and environmental interfaces, so this interface split is a good fit for the item-definition workflow.

Item functions =================
AGENT INSTRUCTIONS

List the high-level functions the item performs.
Each function should be one sentence starting with a verb.
Safety-relevant functions must be clearly marked.
PURPOSE

Functions are the starting point for functional safety requirements and malfunction analysis.
ASIL-D GUIDANCE

Every safety-relevant function must eventually map to at least one safety requirement with an ASIL-D allocation and verification method.
TEMPLATE At the current system-level and variant-generic abstraction, <COMPONENT_NAME> can be described as performing the following functions:

accept supported <REQUEST_TYPE> requests from higher software layers
interpret those requests within the supported <COMPONENT_NAME> software abstraction
determine or apply the required <DECISION_TYPE> within the supported scope
coordinate issuance of lower-boundary requests needed to realize the intended <BEHAVIOR>
return status, completion, or error information back to the requesting software layer
expose supported diagnostics, observability hooks, or health-related reporting allocated to <COMPONENT_NAME>
EXAMPLE (non-Fabric) At the current system-level and variant-generic abstraction, System Health Monitor can be described as performing the following functions:

accept supported health monitoring, polling, or configuration requests from higher software layers
interpret those requests within the supported SHM software abstraction
determine or apply the required thresholds, sampling cadence, or fault conditions within the supported scope
coordinate issuance of lower-boundary requests needed to realize the intended health monitoring behavior
return status, completion, or error information back to the requesting software layer
expose supported diagnostics, observability hooks, or health-related reporting allocated to SHM
Representative item subfunctions ===================================
AGENT INSTRUCTIONS

Decompose the high-level functions into named subfunctions.
Add a reference to the functions detail page if one exists.
Mark subfunctions that are safety-relevant.
PURPOSE

Provides the functional decomposition for requirement derivation and verification planning.
ASIL-D GUIDANCE

Subfunctions that implement safety mechanisms or detect faults must be explicitly identified.
TEMPLATE See <LINK_TO_FUNCTIONS_DETAIL_PAGE> if available.

Representative subfunctions to decompose later may include:

request admission and validation
source / destination interpretation (or equivalent for this item)
<DOMAIN_SPECIFIC_TRANSLATION>
per <UNIT> <DECISION_TYPE> determination
including <SPECIAL_CASE_1>
<ORDERING_OR_INTEGRITY_PROPERTY> preservation
<REQUEST_TYPE> request construction
coordination of multi-step or multi-device <BEHAVIOR> where applicable
completion, timeout, and retry handling where allocated to <COMPONENT_NAME>
error detection, propagation, and reporting behavior
telemetry / monitoring hooks for <METRIC_TYPE> health and status
<METRIC_TYPE> telemetry
<HEALTH_INDICATOR_1>
<HEALTH_INDICATOR_2>
EXAMPLE (non-Fabric) See https://github.com/tenstorrent/tt-fabric-internal/blob/main/safety/shm/functions.md (placeholder) if available.

Representative subfunctions to decompose later may include:

request admission and validation
sensor identity / register interpretation
threshold / hysteresis translation
per sensor anomaly determination
including multi-source aggregation
alarm ordering and priority preservation
fault notification construction
coordination of multi-device health polling where applicable
completion, timeout, and retry handling where allocated to SHM
error detection, propagation, and reporting behavior
telemetry / monitoring hooks for device health and status
temperature telemetry
voltage telemetry
heartbeat (polling activity)
Assumptions of use =====================
AGENT INSTRUCTIONS

List explicit assumptions that must hold for the item to operate safely.
Distinguish assumptions about users, lower layers, hardware, and environment.
Each assumption must be realistic and, where possible, enforceable.
PURPOSE

Assumptions of use are essential for SEooC and are heavily scrutinized at ASIL-D.
ASIL-D GUIDANCE

At ASIL-D, assumptions must be evaluated for plausibility and must be covered by the product safety concept or verified at integration.
TEMPLATE This first-pass item definition assumes:

higher-level software uses supported <COMPONENT_NAME> interfaces correctly
platform and lower-boundary services required by <COMPONENT_NAME> are available and functioning correctly
<CONFIG_TYPE>, connectivity, and addressing information presented to <COMPONENT_NAME> are valid for the intended supported mode
and <CONFIG_TYPE> can be validated (e.g., during <ENUMERATION_PHASE> or any time before <COMPONENT_NAME> initialization)
underlying hardware <HW_CAPABILITY> and <HW_TYPE> mechanisms behave according to their own specifications
system-level supervision and external safety mechanisms exist outside the <COMPONENT_NAME> item boundary where required by the product safety concept
Making assumptions of use explicit is directly aligned with the safety-plan baseline and the existing item-definition structure.

EXAMPLE (non-Fabric) This first-pass item definition assumes:

higher-level software uses supported SHM interfaces correctly
platform and lower-boundary services required by SHM are available and functioning correctly
sensor map, calibration data, and addressing information presented to SHM are valid for the intended supported mode
and sensor map can be validated (e.g., during device enumeration or any time before SHM initialization)
underlying hardware sensors and telemetry mechanisms behave according to their own specifications
system-level supervision and external safety mechanisms exist outside the SHM item boundary where required by the product safety concept
Preconditions =================
AGENT INSTRUCTIONS

List the conditions that must be true before the item can execute safely.
Link each precondition to a code path or initialization sequence.
PURPOSE

Preconditions feed into technical safety requirements and test cases.
TEMPLATE Before <COMPONENT_NAME> execution in a supported mode:

required software stack elements are initialized (<DEPENDENCY_1>, <DEPENDENCY_2>, <DEPENDENCY_3>)
required lower-boundary services are available (<DEPENDENCY_1>, <DEPENDENCY_2>, <DEPENDENCY_3>, device hardware)
participating devices or components are enumerated and reachable within the supported system configuration
required <CONFIG_TYPE>, configuration, and connectivity metadata are available
reset / power / initialization prerequisites for the intended <OPERATION_MODE> are satisfied
<COMPONENT_NAME> reaches defined ready state, globally, before servicing <TRAFFIC_TYPE>
EXAMPLE (non-Fabric) Before System Health Monitor execution in a supported mode:

required software stack elements are initialized (UMD, metal runtime, HAL)
required lower-boundary services are available (UMD, cluster, HAL, device hardware)
participating devices or components are enumerated and reachable within the supported system configuration
required sensor map, configuration, and calibration metadata are available
reset / power / initialization prerequisites for the intended monitoring mode are satisfied
SHM reaches defined ready state, globally, before servicing health requests
Postconditions ==================
AGENT INSTRUCTIONS

Define the observable state after a successful operation.
Define the state after a failed operation (safe state, error reported, etc.).
Clarify the scope of an “operation” (single request, full lifecycle, etc.).
PURPOSE

Postconditions are used to define success criteria and safe-state behavior.
ASIL-D GUIDANCE

At ASIL-D, failed postconditions must lead to a known safe state or to a fault that is propagated and handled.
TEMPLATE After successful execution of a supported <COMPONENT_NAME> operation (define: <OPERATION_SCOPE>):

the intended <REQUEST_TYPE> has been issued or coordinated correctly within the defined boundary; else a reported error produced
completion, status, or error information is available to the caller through the defined path
any relevant monitoring or diagnostic outputs are updated or exposed through the supported interface
<COMPONENT_NAME> internal buffer contents are transient and not reliably inspectable; state may be gone
upon workload completion <COMPONENT_NAME> resources teardown, returning occupied hardware resources to a defined quiescent state such that they can be reused for future <COMPONENT_NAME> (or other workload) launches
EXAMPLE (non-Fabric) After successful execution of a supported SHM operation (a single health query or a monitoring session):

the intended health query or monitoring session has been issued or coordinated correctly within the defined boundary; else a reported error produced
completion, status, or error information is available to the caller through the defined path
any relevant monitoring or diagnostic outputs are updated or exposed through the supported interface
SHM internal buffer contents are transient and not reliably inspectable; state may be gone
upon workload completion SHM resources teardown, returning occupied hardware resources to a defined quiescent state such that they can be reused for future SHM (or other workload) launches
Dependencies ================
AGENT INSTRUCTIONS

List all dependencies (software, hardware, tools, data, environment).
Distinguish dependencies that are safety-relevant from those that are not.
PURPOSE

Dependencies are inputs to the integration plan and dependent failure analysis.
TEMPLATE Key dependencies likely include:

higher software consumers of <COMPONENT_NAME> interfaces
producer to <CONSUMER_TYPE> via APIs
runtime / distributed software components interacting with <COMPONENT_NAME>
lower software layers or driver-facing services used to execute requests
<CONFIG_TYPE> / coordinate / addressing definitions
<HARDWARE_TYPE> or interconnect-related hardware behavior as an environmental dependency
reset, interrupt, memory, and fault-reporting mechanisms where they constrain <COMPONENT_NAME> behavior
EXAMPLE (non-Fabric) Key dependencies likely include:

higher software consumers of SHM interfaces
producer to runtime / safety manager via APIs
runtime / distributed software components interacting with SHM
lower software layers or driver-facing services used to execute requests
sensor map / coordinate / addressing definitions
thermal / voltage sensor hardware behavior as an environmental dependency
reset, interrupt, memory, and fault-reporting mechanisms where they constrain SHM behavior
Preliminary safety-related failure considerations =====================================================
AGENT INSTRUCTIONS

Brainstorm failure modes / malfunctions grouped by theme.
Use ISO 26262-style malfunction language (e.g., “wrong X”, “unintended X”, “loss of X”, “too early/late X”).
Do not attempt to assign ASILs yet; that comes from hazard analysis.
PURPOSE

Provides the starting point for the malfunction analysis and FMEA/FMEDA.
ASIL-D GUIDANCE

At ASIL-D, the failure list must be comprehensive; use structured guidance (HARA, STPA, FMEA) to avoid gaps.
TEMPLATE Potential failure themes to analyze later include:

wrong <TARGET> device, endpoint, or <PEER_TYPE> selection
wrong <DECISION_TYPE>, <CONFIG_TYPE> interpretation, or connectivity assumption
<REQUEST_TYPE> not issued when required
<REQUEST_TYPE> issued incorrectly, incompletely, to the wrong destination, or delivered in the wrong order
timeout, completion, or fault status reported incorrectly
loss of fault propagation from lower layers into higher software
mismatch between supported <COMPONENT_NAME> abstraction and actual <CONFIG_TYPE> constraints in a given variant
stale, inconsistent, or incorrect <CONFIG_TYPE> / descriptor data leading to unsafe behavior
incorrect state reported: telemetry, <COMPONENT_NAME> state, error conditions
insufficient performance relative to SLAs
This is consistent with the baseline safety process of using item definitions as the basis for later requirements, malfunction analysis, and verification planning.

EXAMPLE (non-Fabric) Potential failure themes to analyze later include:

wrong sensor, endpoint, or health domain selection
wrong threshold, calibration interpretation, or connectivity assumption
health query not issued when required
alarm issued incorrectly, incompletely, to the wrong destination, or delivered in the wrong order
timeout, completion, or fault status reported incorrectly
loss of fault propagation from lower layers into higher software
mismatch between supported SHM abstraction and actual sensor constraints in a given variant
stale, inconsistent, or incorrect sensor map / calibration data leading to unsafe behavior
incorrect state reported: telemetry, SHM state, error conditions
insufficient performance relative to SLAs (e.g., FTTI)
Preliminary work products to derive next ==============================================
AGENT INSTRUCTIONS

List the safety work products that should be created next.
Use the program’s naming conventions.
PURPOSE

Shows the path from item definition to full safety case.
TEMPLATE Recommended follow-on work products:

detailed <COMPONENT_NAME> boundary diagram (e.g., <COMP>-item-boundary.md)
layered context diagram showing <COMPONENT_NAME> relative to higher software, lower services, and hardware/interconnect environment
boundary diagrams + interface specification (e.g., <COMP>-boundary-and-interfaces.md)
interface specification for upward, downward, and environmental interfaces
assumptions-of-use record (e.g., <COMP>-assumptions-pre-post-deps.md)
functional decomposition with named subfunctions and ownership (e.g., <COMP>-functions.md)
preliminary malfunction list (e.g., <COMP>-failure-considerations.md)
high-level requirements split into:
<COMP>-safety-properties.md (negative / invariant requirements)
<COMP>-functional-capabilities.md (positive / measurable requirements + envelope)
first-cut safety requirements allocation proposal
verification strategy for safety-relevant <COMPONENT_NAME> behavior
baselining decision record (e.g., <COMP>-baseline-decisions.md) for §15 open items
This sequencing matches the safety-plan flow from scope and boundary definition into requirements, design, and verification work products.

EXAMPLE (non-Fabric) Recommended follow-on work products:

detailed System Health Monitor boundary diagram (SHM-item-boundary.md)
layered context diagram showing SHM relative to higher software, lower services, and hardware environment
boundary diagrams + interface specification (SHM-boundary-and-interfaces.md)
interface specification for upward, downward, and environmental interfaces
assumptions-of-use record (SHM-assumptions-pre-post-deps.md)
functional decomposition with named subfunctions and ownership (SHM-functions.md)
preliminary malfunction list (SHM-failure-considerations.md)
high-level requirements split into:
SHM-safety-properties.md (negative / invariant requirements)
SHM-functional-capabilities.md (positive / measurable requirements + envelope)
first-cut safety requirements allocation proposal
verification strategy for safety-relevant SHM behavior
baselining decision record (SHM-baseline-decisions.md) for §15 open items
Known gaps / open questions ===============================
AGENT INSTRUCTIONS

List open questions that must be resolved before baselining the item definition.
Assign owners and target dates if possible.
Track the resolution of open decisions in a companion file (e.g., <COMP>-baseline-decisions.md) and update this section to reference it.
PURPOSE

Tracks closure of open items and records decisions made during the §15 baselining session.
TEMPLATE The following points should be confirmed before baselining this page:

the precise software decomposition of <COMPONENT_NAME>
whether <NEIGHBOR_DOMAIN> or <OTHER_DOMAIN> owns any of the currently proposed <COMPONENT_NAME> functions
the exact lower-boundary interface path
what <CONFIG_TYPE>, addressing, and route concepts are explicitly part of <COMPONENT_NAME>
whether retry, timeout, flow-control, and fault-handling behaviors belong in <COMPONENT_NAME> or below it
which diagnostics and observability hooks are owned by <COMPONENT_NAME>
which assumptions must remain generic across variants versus moved into customer-specific tailoring
EXAMPLE (non-Fabric) The following points should be confirmed before baselining this page:

the precise software decomposition of System Health Monitor
whether Runtime or Safety Manager owns any of the currently proposed SHM functions
the exact lower-boundary interface path
what sensor map, addressing, and threshold concepts are explicitly part of SHM
whether retry, timeout, flow-control, and fault-handling behaviors belong in SHM or below it
which diagnostics and observability hooks are owned by SHM
which assumptions must remain generic across variants versus moved into customer-specific tailoring
UML source ==============
AGENT INSTRUCTIONS

Provide PlantUML (or other) diagrams for boundary, internal function, layered component, and functional flow.
Replace placeholder names with the component name and actual function names.
Keep the boundary diagram separate from the internal function view.
Diagrams are appreciated where they make sense. Include at least a boundary / context diagram; add internal function, layered component, and flow diagrams when they clarify ownership, interfaces, or behavior. Skip diagrams that would be redundant or empty.
PURPOSE

Visual models support review and traceability.
PlantUML A: Boundary / context diagram
@startuml skinparam componentStyle rectangle skinparam shadowing false skinparam packageStyle rectangle skinparam defaultTextAlignment center

title <COMPONENT_NAME> Boundary / Context Diagram

rectangle “Higher Software /\nApplication / Runtime Policy /\nDistributed Control” as Higher

package “<COMPONENT_NAME> Item Boundary” <<Rectangle>> { rectangle “<COMPONENT_NAME> software behavior\n- <BEHAVIOR_1>\n- <BEHAVIOR_2>\n- <BEHAVIOR_3>\n- <REQUEST_CONSTRUCTION>\n- <REPORTING_BEHAVIOR>” as Item rectangle “Diagnostics / Status / Telemetry” as Diag }

rectangle “Kernel / Driver /\nLower Boundary Services” as Lower rectangle “<HARDWARE_TYPE> / Interconnect /\nDevice Connectivity HW” as Conn rectangle “External safety mechanisms /\nSystem-level supervision” as Safety rectangle “Customer integration /\nApplication policy outside scope” as Outside

Higher --> Item : supported requests / control intent Item --> Diag : status / telemetry Item --> Lower : lower-boundary services Lower --> Conn : execution path Safety …> Item : external supervision / assumptions Outside …> Item : excluded policy / integration context

@enduml

To show the functions in a UML-style diagram, it is better to create a separate internal functional decomposition view rather than overloading the boundary diagram.

PlantUML A1: <COMPONENT_NAME> internal function view
@startuml skinparam componentStyle rectangle skinparam shadowing false skinparam packageStyle rectangle skinparam defaultTextAlignment center

title <COMPONENT_NAME> Internal Function View

rectangle “Higher Software /\nRuntime / Distributed” as Higher rectangle “Lower Boundary Services” as Lower rectangle “<CONFIG_TYPE> / Connectivity\nEnvironment” as Env rectangle “Diagnostics / Status /\nTelemetry Consumers” as Diag

package “<COMPONENT_NAME>” { rectangle “Request Admission /\nValidation” as F1 rectangle “Target / Endpoint\nInterpretation” as F2 rectangle “<DOMAIN> / <DECISION>\nInterpretation” as F3 rectangle “<REQUEST_TYPE> /\nRequest Construction” as F4 rectangle “Coordination / Dispatch” as F5 rectangle “Completion / Fault\nStatus Handling” as F6 rectangle “Telemetry / Observability\nReporting” as F7 }

Higher --> F1 : request F1 --> F2 : accepted request F2 --> F3 : target context Env --> F3 : <CONFIG_TYPE> / connectivity data F3 --> F4 : <DECISION_CONTEXT> F4 --> F5 : executable request F5 --> Lower : lower-boundary request Lower --> F6 : completion / error / fault F6 --> Higher : result / error status F6 --> F7 : status / event data F7 --> Diag : telemetry / diagnostics

@enduml

PlantUML B: Layered component / interface view
@startuml skinparam componentStyle rectangle skinparam shadowing false skinparam defaultTextAlignment center

title <COMPONENT_NAME> Layered Component / Interface View

together { rectangle “Runtime /\nOrchestration” as Runtime rectangle “Distributed Control /\nCoordination” as Dist rectangle “Other SW Clients /\nFuture Users” as Other }

rectangle “<COMPONENT_NAME>\n\n- <BEHAVIOR_1>\n- <BEHAVIOR_2>\n- <COORDINATION_BEHAVIOR>\n- <DISPATCH_BEHAVIOR>” as Item

together { rectangle “Kernel / Driver Services /\nTransport Enablers” as Kernel rectangle “Platform / Mapping /\nEnumeration Services” as Platform }

together { rectangle “<ON_CHIP_HW> / On-chip Fabric HW /\nRouting Environment” as Noc rectangle “Inter-device Connectivity /\nTransport Environment” as InterDev }

Runtime --> Item : supported APIs / requests Dist --> Item : supported APIs / requests Other --> Item : supported APIs / requests

Item --> Kernel : lower-boundary service requests Item --> Platform : <CONFIG_TYPE> / mapping / discovery dependencies

Kernel --> Noc : execution path Platform --> Noc : environment / mapping Kernel --> InterDev : execution path Platform --> InterDev : environment / mapping

@enduml

PlantUML C: Functional flow / swimlane view
@startuml skinparam shadowing false skinparam sequenceMessageAlign center skinparam responseMessageBelowArrow true

title <COMPONENT_NAME> Functional Flow / Swimlane View

participant “Higher SW Client” as Higher participant “<COMPONENT_NAME>” as Item participant “Lower SW Boundary” as Lower participant “<HARDWARE_TYPE> Environment\n(<ON_CHIP_HW> / Inter-device)” as Conn

Higher -> Item : create request activate Item Item -> Item : validate / accept request Item -> Item : interpret target / <CONFIG_TYPE> / <DECISION> Item -> Lower : prepare / issue lower service request activate Lower Lower -> Conn : execute <CONNECTIVITY_OPERATION> activate Conn Conn --> Lower : completion / fault indication deactivate Conn Lower --> Item : status / fault deactivate Lower Item -> Item : collect completion / fault status Item --> Higher : return result / error / telemetry deactivate Item

@enduml

Detailed safety requirements ================================
AGENT INSTRUCTIONS

Produce two companion files:
<COMP>-safety-properties.md — what must never go wrong (negative / invariant).
<COMP>-functional-capabilities.md — what the item must do, plus performance and supported envelope (positive / measurable).
Derive both from the code, the functions in Sections 7–8, the failure considerations in Section 13, and the interfaces in Section 6.
Do not copy implementation details verbatim; state intent and the safety property or capability to be enforced.
Every requirement must be traceable to a code artifact, a function, a failure theme, or a product requirement.
PURPOSE

Produces the first-cut safety requirements that will be refined into the full safety requirements specification.
ASIL-D GUIDANCE

At ASIL-D, each safety requirement must be unambiguous, verifiable, internally consistent, and free from contradictions with other requirements.
Requirements must be allocated to the item or to a neighboring item / external safety mechanism.
Verification methods must cover fault injection, review, analysis, and testing as appropriate. Independence rules (ISO 26262-6 Table 1) must be respected.
17.1 Requirement split
Safety properties (negative / invariant): what the item must never do wrong. These derive from §13 failure themes and are checked by properties, lint rules, and fault-injection tests.
Functional capabilities (positive / measurable): what the item must do, plus performance targets and the supported envelope (scale / topology / configuration limits). These derive from product requirements and are checked by feature tests and benchmarks.
Where the two overlap, the safety-properties file is the authority and the functional file cross-links to it.
17.2 Provenance and status conventions
Use the following provenance tags on every requirement:

SPEC:<id> — backed by a formal spec/invariant (strongest; independent of code).
CODE:<path> — observed in the implementation; confirm it is intended, not incidental.
HAZARD:§13 — derived top-down from a failure theme; may be a gap today.
PRODUCT — authoritative product requirement.
Use the following status tags:

FIRM — spec-backed, ready to baseline.
CANDIDATE — code-derived, confirm intent and ownership.
PROPOSED — hazard-derived, decide ownership and implementation.
17.3 Safety properties template
SG-<X> — <Safety goal title>
Goal: <one-sentence goal statement>.

<COMP>-HLR-<nn>a — <Short requirement name>. <COMPONENT_NAME> shall <NEGATIVE_OR_INVARIANT_BEHAVIOR>. <PROVENANCE> · <STATUS>

<COMP>-HLR-<nn>b — <Short requirement name>. <COMPONENT_NAME> shall <NEGATIVE_OR_INVARIANT_BEHAVIOR>. <PROVENANCE> · <PROVENANCE> · <STATUS>

Optional decision / note block.

Example (non-Fabric): SHM safety properties
SG-A — Health indicator integrity
Goal: every configured health indicator is sampled and reported correctly, or a fault is raised.

SHM-HLR-01a — No missed sample. SHM shall not miss a configured periodic health-indicator sample without reporting a fault. CODE:shm_polling.cpp · HAZARD:§13 · CANDIDATE

SHM-HLR-01b — No stale sample. SHM shall not report a health-indicator value older than its sampling period without flagging it stale. CODE:shm_buffer.hpp · HAZARD:§13 · CANDIDATE

SG-B — Fault propagation
Goal: detected anomalies are reported to the safety manager within the FTTI.

SHM-HLR-02a — Report within FTTI. SHM shall report a detected out-of-range health indicator to the safety manager within the configured FTTI. SPEC:SHM-FTTI-01 · HAZARD:§13 · FIRM

SHM-HLR-02b — No silent fault. SHM shall not suppress a health-indicator fault due to a lower-layer sampling error. HAZARD:§13 · PROPOSED

17.4 Functional capabilities template
Capability requirements (FR)
<COMP>-FR-<nn> — <Capability name>. <COMPONENT_NAME> shall <MEASURABLE_CAPABILITY>. <PROVENANCE>

Performance requirements (PERF)
<COMP>-PERF-<nn> — <Performance name>. <COMPONENT_NAME> shall provide <MEASURABLE_TARGET>. <PROVENANCE> (Non-functional / performance — verification is benchmark-based, not property-based.)

Supported envelope (ENV)
<COMP>-ENV-<nn> — <Envelope name>. Within the supported envelope, <COMPONENT_NAME> shall support <SCALE_OR_TOPOLOGY_LIMIT>. <PROVENANCE>

The envelope defines the scope of “every supported configuration” used in the safety requirements (e.g., <COMP>-HLR-<nn> is required across this envelope).

Example (non-Fabric): SHM functional capabilities
SHM-FR-01 — Any sensor query. SHM shall support querying the health status of any configured sensor from any host or device client. PRODUCT

SHM-FR-02 — Threshold configurability. SHM shall support per-sensor configurable high and low thresholds. PRODUCT

SHM-PERF-01 — Sampling latency. SHM shall sample all configured sensors and make results available within 10 ms. PRODUCT

SHM-ENV-01 — Scale limits. Within the supported envelope: max 64 sensors, max 16 devices. PRODUCT

The envelope defines the scope of “every supported configuration” used in the safety requirements (e.g., SHM-HLR-01a is required across this envelope).

17.5 Baselining status checklist
AGENT INSTRUCTIONS

Track the completion of the safety work products.
Update the checklist as items move from DRAFT to baseline.
TEMPLATE

[ ] Step A — high-level requirements (<COMP>-safety-properties.md, <COMP>-functional-capabilities.md)
[ ] Step B — item boundary (document / diagram)
[ ] Step C — boundary diagrams + interface specification
[ ] §15 baselining session — owner decisions resolved
[ ] Step D — functions + subfunctions + requirement allocation matrix
[ ] Step E — preliminary failure / malfunction list
[ ] Step F — assumptions / preconditions / postconditions / dependencies
EXAMPLE (non-Fabric)

[x] Step A — high-level requirements (SHM-safety-properties.md, SHM-functional-capabilities.md)
[ ] Step B — item boundary
[ ] Step C — boundary diagrams + interface specification
[ ] §15 baselining session — owner decisions resolved
[ ] Step D — functions + subfunctions + requirement allocation matrix
[ ] Step E — preliminary failure / malfunction list
[ ] Step F — assumptions / preconditions / postconditions / dependencies
Sources
Item Definition Template
AI-IP SW Safety Program: ISO 26262 ASIL-D SEooC Management Approach and ASPICE Pilot Execution
<Component-specific architecture and source code>
